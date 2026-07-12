#include "../common/PCFG.h"
#include "../common/md5.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>

using namespace std;
using namespace chrono;

// ============================================================
//  Serial 版本入口
//  流程：训练 → 生成猜测 → MD5 哈希 → 输出统计
//  作为所有并行版本的性能对比基线（baseline）
// ============================================================
int main(int argc, char *argv[])
{
    string train_path = "rockyou.txt";
    if (argc > 1) train_path = argv[1];

    double time_train = 0, time_guess = 0, time_hash = 0;
    PriorityQueue q;

    // ---- 1. 训练模型 ----
    auto t0 = system_clock::now();
    q.m.train(train_path);
    q.m.order();
    auto t1 = system_clock::now();
    time_train = duration<double>(t1 - t0).count();
    cout << "Training completed in " << time_train << " s" << endl;

    // ---- 2. 初始化优先队列 ----
    q.init();
    cout << "Priority queue initialized." << endl;

    // ---- 3. 猜测生成 + MD5 哈希 ----
    int curr_num = 0, history = 0;
    auto start = system_clock::now();

    while (!q.priority.empty())
    {
        q.PopNext();
        q.total_guesses = q.guesses.size();

        if (q.total_guesses - curr_num >= 100000)
        {
            cout << "Guesses generated: " << history + q.total_guesses << endl;
            curr_num = q.total_guesses;

            if (history + q.total_guesses > 10000000)
            {
                auto end = system_clock::now();
                time_guess = duration<double>(end - start).count();
                cout << "Guess time:  " << time_guess - time_hash << " s" << endl;
                cout << "Hash time:   " << time_hash  << " s" << endl;
                cout << "Train time:  " << time_train << " s" << endl;
                break;
            }
        }

        // 每攒够约 100 万条猜测做一次 MD5 批量哈希
        if (curr_num > 1000000)
        {
            auto hs = system_clock::now();
            for (string pw : q.guesses)
            {
                bit32 state[4];
                MD5Hash_serial(pw, state);
                // 实际应用中此处会与目标哈希比对
            }
            auto he = system_clock::now();
            time_hash += duration<double>(he - hs).count();
            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }
    }

    cout << "\n=== Serial Benchmark Summary ===" << endl;
    cout << "Total training time: " << time_train  << " s" << endl;
    cout << "Total guess time:    " << time_guess  << " s" << endl;
    cout << "Total hash time:     " << time_hash   << " s" << endl;

    return 0;
}
