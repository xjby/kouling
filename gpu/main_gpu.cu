#include "PCFG.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdlib>

using namespace std;
using namespace chrono;

// ============================================================
//  GPU (CUDA) 版本入口
//
//  架构：
//    - Host (CPU) 负责模型训练、优先队列管理
//    - Device (GPU) 负责大规模并行生成猜测
//    - 每次 Generate 调用 CUDA kernel，结果拷回 Host
//
//  内存传输模式：
//    Host → Device: segment values (FlatStr), prefix string
//    Device → Host: 生成的猜测字符串数组
//
//  编译（nvcc）：
//    nvcc -O3 -arch=sm_60 main_gpu.cu guessing_gpu.cu train.cu md5.cu -o main_gpu
// ============================================================

extern void GPU_Init();

int main(int argc, char *argv[])
{
    GPU_Init();

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
    cout << "[GPU] Training completed in " << time_train << " s" << endl;

    // ---- 2. 初始化优先队列 ----
    q.init();
    cout << "[GPU] Priority queue initialized." << endl;

    // ---- 3. GPU 加速猜测生成 ----
    int curr_num = 0, history = 0;
    auto start = system_clock::now();

    while (!q.priority.empty())
    {
        q.PopNext_gpu();    // ★ GPU 版：每次弹出用 CUDA kernel 批量生成
        q.total_guesses = q.guesses.size();

        if (q.total_guesses - curr_num >= 10000)
        {
            cout << "[GPU] Guesses generated: " << history + q.total_guesses << endl;
            curr_num = q.total_guesses;

            if (history + q.total_guesses > 500000)
            {
                auto end = system_clock::now();
                time_guess = duration<double>(end - start).count();
                cout << "[GPU] Guess time: " << time_guess - time_hash << " s" << endl;
                cout << "[GPU] Hash time:  " << time_hash  << " s" << endl;
                cout << "[GPU] Train time: " << time_train << " s" << endl;
                break;
            }
        }

        if (curr_num > 50000)
        {
            auto hs = system_clock::now();
            for (string pw : q.guesses)
            {
                bit32 state[4];
                extern void MD5Hash_serial(string, bit32*);
                MD5Hash_serial(pw, state);
            }
            auto he = system_clock::now();
            time_hash += duration<double>(he - hs).count();
            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }
    }

    cout << "\n=== GPU Benchmark Summary ===" << endl;
    cout << "Total training time: " << time_train  << " s" << endl;
    cout << "Total guess time:    " << time_guess  << " s" << endl;
    cout << "Total hash time:     " << time_hash   << " s" << endl;

    return 0;
}
