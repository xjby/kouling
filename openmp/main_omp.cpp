#include "../common/PCFG.h"
#include "../common/md5.h"
#include <omp.h>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>

using namespace std;
using namespace chrono;

// ============================================================
//  OpenMP 版本入口 — 多核 CPU 并行
//
//  两级并行：
//    Level 1 — 口令猜测生成并行化
//      Generate() 内循环用 #pragma omp parallel for
//      每个线程负责一段 segment value 子集
//
//    Level 2 — MD5 哈希计算并行化
//      guesses 数组拆分给多个线程，
//      每线程独立计算自己那部分口令的 MD5
//
//  环境变量：
//    export OMP_NUM_THREADS=4   # 设置线程数
//    export OMP_PROC_BIND=true  # 绑定线程到核心
//
//  编译：
//    g++ -fopenmp -O3 ...
// ============================================================
int main(int argc, char *argv[])
{
    string train_path = "rockyou.txt";
    if (argc > 1) train_path = argv[1];

    int num_threads = omp_get_max_threads();
    cout << "[OpenMP] Using " << num_threads << " threads" << endl;

    double time_train = 0, time_guess = 0, time_hash = 0;
    PriorityQueue q;

    // ---- 1. 训练 ----
    auto t0 = system_clock::now();
    q.m.train(train_path);
    q.m.order();
    auto t1 = system_clock::now();
    time_train = duration<double>(t1 - t0).count();
    cout << "[OpenMP] Training completed in " << time_train << " s" << endl;

    // ---- 2. 初始化 ----
    q.init();
    cout << "[OpenMP] Priority queue initialized." << endl;

    // ---- 3. 猜测生成 + 并行 MD5 ----
    int curr_num = 0, history = 0;
    auto start = system_clock::now();

    while (!q.priority.empty())
    {
        q.PopNext();
        q.total_guesses = q.guesses.size();

        if (q.total_guesses - curr_num >= 100000)
        {
            cout << "[OpenMP] Guesses generated: " << history + q.total_guesses << endl;
            curr_num = q.total_guesses;

            if (history + q.total_guesses > 10000000)
            {
                auto end = system_clock::now();
                time_guess = duration<double>(end - start).count();
                cout << "[OpenMP] Guess time: " << time_guess - time_hash << " s" << endl;
                cout << "[OpenMP] Hash time:  " << time_hash  << " s" << endl;
                cout << "[OpenMP] Train time: " << time_train << " s" << endl;
                break;
            }
        }

        if (curr_num > 1000000)
        {
            auto hs = system_clock::now();
            int n = q.guesses.size();

            // ★ OpenMP 并行 MD5 哈希：按线程拆分 guesses
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; i++)
            {
                bit32 state[4];
                MD5Hash_serial(q.guesses[i], state);
                // 实际应用中此处与目标哈希比对
            }

            auto he = system_clock::now();
            time_hash += duration<double>(he - hs).count();
            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }
    }

    cout << "\n=== OpenMP Benchmark Summary (" << num_threads << " threads) ===" << endl;
    cout << "Total training time: " << time_train  << " s" << endl;
    cout << "Total guess time:    " << time_guess  << " s" << endl;
    cout << "Total hash time:     " << time_hash   << " s" << endl;

    return 0;
}
