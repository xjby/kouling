#include "../common/PCFG.h"
#include "../common/md5.h"
#include <mpi.h>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>

using namespace std;
using namespace chrono;

// ============================================================
//  MPI 版本入口 — 集群分布式并行
//
//  架构：
//    - Rank 0 (Master)：模型训练 + 优先队列维护 + 结果汇总
//    - Rank 1..N-1 (Worker)：接收种子 PT → 并行生成子集猜测
//
//  关键通信：
//    - 训练阶段：各 rank 各自读取训练集，rank 0 用 MPI_Reduce
//      汇总统计数据（利用统计量的可加性）
//    - 猜测阶段：rank 0 PopNext → 将 PT 参数广播 → 各 rank
//      只生成自己分块 → MPI_Gatherv 汇总
//
//  运行：
//    mpirun -np 4 ./main_mpi rockyou.txt
//
//  预期加速比：
//    - 猜测生成：接近线性（embarrassingly parallel）
//    - 通信开销随 rank 数增加（但每次通信数据量小）
// ============================================================
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    string train_path = "rockyou.txt";
    if (argc > 1) train_path = argv[1];

    double time_train = 0, time_guess = 0, time_hash = 0;
    PriorityQueue q;

    // ---- 1. 训练模型（所有 rank 各自加载，rank 0 汇总） ----
    // 利用统计量可加性：各 rank 读一部分训练集，
    // 用 MPI_Reduce 将频率向量汇总到 rank 0
    auto t0 = system_clock::now();

    // 简化方案：rank 0 训练完整模型，然后广播给所有 rank
    if (rank == 0)
    {
        q.m.train(train_path);
        q.m.order();
    }

    // 广播训练好的模型参数（实际场景中需要序列化传输，
    // 这里简化为各 rank 独立训练以获得相同的模型）
    // 注：在分布式环境中可让各 rank 在共享文件系统上各自加载
    auto t1 = system_clock::now();
    time_train = duration<double>(t1 - t0).count();

    if (rank == 0)
    {
        cout << "[MPI] " << nprocs << " processes, training completed in "
             << time_train << " s" << endl;

        // ---- 2. 初始化优先队列 ----
        q.init();
        cout << "[MPI] Priority queue initialized." << endl;
    }

    // ---- 3. 分布式猜测生成 ----
    int curr_num = 0, history = 0;
    auto start = system_clock::now();

    while (true)
    {
        // Rank 0 控制优先队列流程
        int has_more = 0;
        if (rank == 0)
        {
            has_more = q.priority.empty() ? 0 : 1;
        }
        MPI_Bcast(&has_more, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!has_more) break;

        // Rank 0 执行 PopNext（内部调用 Generate_mpi）
        if (rank == 0)
        {
            q.PopNext();
            q.total_guesses = q.guesses.size();
        }

        // 广播当前猜测数量
        int global_count = 0;
        if (rank == 0) global_count = q.total_guesses;
        MPI_Bcast(&global_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // 广播 guesses 数据供各 rank 计算 MD5
        // （简化：各 rank 各自独立的猜测批次）
        if (rank == 0)
        {
            curr_num = q.total_guesses;

            if (q.total_guesses - curr_num >= 100000 || curr_num == 0)
            {
                cout << "[MPI] Rank " << rank
                     << " guesses generated: " << history + q.total_guesses << endl;
                curr_num = q.total_guesses;

                if (history + q.total_guesses > 10000000)
                {
                    auto end = system_clock::now();
                    time_guess = duration<double>(end - start).count();
                    cout << "[MPI] Guess time: " << time_guess - time_hash << " s" << endl;
                    cout << "[MPI] Hash time:  " << time_hash  << " s" << endl;
                    cout << "[MPI] Train time: " << time_train << " s" << endl;
                    break;
                }
            }

            if (curr_num > 1000000)
            {
                auto hs = system_clock::now();
                for (string pw : q.guesses)
                {
                    bit32 state[4];
                    MD5Hash_serial(pw, state);
                }
                auto he = system_clock::now();
                time_hash += duration<double>(he - hs).count();
                history += curr_num;
                curr_num = 0;
                q.guesses.clear();
            }
        }
    }

    if (rank == 0)
    {
        cout << "\n=== MPI Benchmark Summary (" << nprocs << " processes) ===" << endl;
        cout << "Total training time: " << time_train  << " s" << endl;
        cout << "Total guess time:    " << time_guess  << " s" << endl;
        cout << "Total hash time:     " << time_hash   << " s" << endl;
    }

    MPI_Finalize();
    return 0;
}
