#include "../common/PCFG.h"
#include "../common/md5.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>

using namespace std;
using namespace chrono;

// ============================================================
//  SIMD (ARM NEON) 版本入口
//
//  架构特点：
//    - ARM NEON 128-bit SIMD 指令集
//    - 4 路数据并行：一次处理 4 条口令的 MD5 哈希
//    - 猜测生成仍为串行（热点在 MD5 哈希阶段）
//
//  关键优化：
//    - F/G/H/I → 对应 NEON 向量指令
//    - uint32x4_t 同时持有 4 条口令的状态字
//    - 批量 4 条对齐，减少循环开销
//
//  目标平台：ARM AArch64 (树莓派、鲲鹏等)
//  编译：g++ -march=armv8-a+simd -O3 ...
// ============================================================
int main(int argc, char *argv[])
{
    string train_path = "rockyou.txt";
    if (argc > 1) train_path = argv[1];

    double time_train = 0, time_guess = 0, time_hash = 0;
    PriorityQueue q;

    // ---- 1. 训练（同 serial） ----
    auto t0 = system_clock::now();
    q.m.train(train_path);
    q.m.order();
    auto t1 = system_clock::now();
    time_train = duration<double>(t1 - t0).count();
    cout << "[NEON] Training completed in " << time_train << " s" << endl;

    // ---- 2. 初始化优先队列 ----
    q.init();
    cout << "[NEON] Priority queue initialized." << endl;

    // ---- 3. 猜测生成 + NEON 批量 MD5 ----
    int curr_num = 0, history = 0;
    auto start = system_clock::now();

    while (!q.priority.empty())
    {
        q.PopNext();
        q.total_guesses = q.guesses.size();

        if (q.total_guesses - curr_num >= 100000)
        {
            cout << "[NEON] Guesses generated: " << history + q.total_guesses << endl;
            curr_num = q.total_guesses;

            if (history + q.total_guesses > 10000000)
            {
                auto end = system_clock::now();
                time_guess = duration<double>(end - start).count();
                cout << "[NEON] Guess time: " << time_guess - time_hash << " s" << endl;
                cout << "[NEON] Hash time:  " << time_hash  << " s" << endl;
                cout << "[NEON] Train time: " << time_train << " s" << endl;
                break;
            }
        }

        // ★ NEON 核心优化：每次取 4 条口令，用 SIMD 批量计算哈希
        if (curr_num > 1000000)
        {
            auto hs = system_clock::now();
            int n = q.guesses.size();
            string batch[4];
            bit32 states[4][4];
            int batch_idx = 0;

            for (string pw : q.guesses)
            {
                batch[batch_idx++] = pw;
                if (batch_idx == 4)
                {
                    MD5Hash_4x(batch, states);   // ★ NEON 4路并行哈希
                    batch_idx = 0;
                }
            }
            if (batch_idx > 0)
            {
                for (int i = batch_idx; i < 4; i++) batch[i] = "";
                MD5Hash_4x(batch, states);
            }
            auto he = system_clock::now();
            time_hash += duration<double>(he - hs).count();
            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }
    }

    cout << "\n=== SIMD (NEON) Benchmark Summary ===" << endl;
    cout << "Total training time: " << time_train  << " s" << endl;
    cout << "Total guess time:    " << time_guess  << " s" << endl;
    cout << "Total hash time:     " << time_hash   << " s" << endl;

    return 0;
}
