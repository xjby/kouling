#include "PCFG.h"
#include <chrono>
#include <fstream>
#include <sstream>
#include "md5.h"
#include <iomanip>
#include <hip/hip_runtime.h>
using namespace std;
using namespace chrono;

extern void GPU_Init();

int main()
{
    GPU_Init();

    cout << "Testing MD5Hash correctness..." << endl;
    string test_pws[8] = {"123456", "password", "12345678", "qwerty", "123456789", "12345", "1234", "111111"};
    string test_hashes[8] = {
        "e10adc3949ba59abbe56e057f20f883e", "5f4dcc3b5aa765d61d8327deb882cf99",
        "25d55ad283aa400af464c76d713c07ad", "d8578edf8458ce06fbc5bb76a58c5ca4",
        "25f9e794323b453885f5181f1b624d0b", "827ccb0eea8a706c4c34a16891f84e7b",
        "81dc9bdb52d04dc20036dbd8313ed055", "96e79218965eb72c92a549dd5a330112"
    };
    bit32 states[4][4];
    for (int i = 0; i < 8; i += 4) {
        string batch[4] = {test_pws[i], test_pws[i+1], test_pws[i+2], test_pws[i+3]};
        MD5Hash(batch, states);
        for (int k = 0; k < 4; k++) {
            stringstream ss;
            for (int i1 = 0; i1 < 4; i1++)
                ss << setw(8) << setfill('0') << hex << states[k][i1];
            if (ss.str() != test_hashes[i+k]) {
                cout << "MD5Hash test failed for " << test_pws[i+k] << "!" << endl;
                return 1;
            }
        }
    }
    cout << "MD5Hash test passed!" << endl;

    double time_hash = 0, time_guess = 0, time_train = 0;
    PriorityQueue q;

    auto t0 = system_clock::now();
    q.m.train("rockyou.txt");
    q.m.order();
    auto t1 = system_clock::now();
    auto dur = duration_cast<microseconds>(t1 - t0);
    time_train = double(dur.count()) * microseconds::period::num / microseconds::period::den;

    q.init();
    cout << "here" << endl;

    int curr_num = 0, history = 0;
    auto start = system_clock::now();

    while (!q.priority.empty()) {
        q.PopNext();
        q.total_guesses = q.guesses.size();

        if (q.total_guesses - curr_num >= 10000) {
            cout << "Guesses generated: " << history + q.total_guesses << endl;
            curr_num = q.total_guesses;
            if (history + q.total_guesses > 500000) {
                auto end = system_clock::now();
                auto d = duration_cast<microseconds>(end - start);
                time_guess = double(d.count()) * microseconds::period::num / microseconds::period::den;
                cout << "Guess time:" << time_guess - time_hash << "seconds" << endl;
                cout << "Hash time:" << time_hash << "seconds" << endl;
                cout << "Train time:" << time_train << "seconds" << endl;
                break;
            }
        }

        if (curr_num > 50000) {
            auto hs = system_clock::now();
            int count = 0;
            string batch[4];
            bit32 states2[4][4];
            for (string pw : q.guesses) {
                batch[count++] = pw;
                if (count == 4) { MD5Hash(batch, states2); count = 0; }
            }
            if (count > 0) {
                for (int i2 = count; i2 < 4; i2++) batch[i2] = "";
                MD5Hash(batch, states2);
            }
            auto he = system_clock::now();
            auto d2 = duration_cast<microseconds>(he - hs);
            time_hash += double(d2.count()) * microseconds::period::num / microseconds::period::den;
            history += curr_num;
            curr_num = 0;
            q.guesses.clear();
        }
    }
}
