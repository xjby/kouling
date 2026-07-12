#include "PCFG.h"
#include <vector>
#include <cstring>
#include <iostream>
#include <cuda_runtime.h>
using namespace std;

// ============================================================
//  GPU (CUDA) 版本 — 口令猜测生成
//
//  并行化策略：
//    将 Generate() 最内层的 for 循环映射为 CUDA kernel，
//    每个线程负责生成一条猜测。
//
//  两个 kernel：
//    kernel_single  — 单 segment PT（无前缀拼接）
//    kernel_multi   — 多 segment PT（前缀 + 后缀拼接）
//
//  FlatStr 辅助结构：扁平化存储变长字符串
//    - d_data: 所有 value 拼接为连续字符数组
//    - d_off:   每条 value 在 d_data 中的起始偏移
//    - d_len:   每条 value 的长度
//
//  线程配置：
//    - 256 threads/block
//    - grid = ceil(total / 256)
// ============================================================

#define MAX_PW_LEN 128
#define THREADS 256

// ---- FlatStr: 将 vector<string> 扁平化为 GPU 可访问的格式 ----
struct FlatStr {
    char *d_data;
    int *d_off, *d_len;
    int total_chars, n;

    void alloc(const vector<string>& v) {
        n = v.size();
        vector<int> off(n), len(n);
        total_chars = 0;
        for (int i = 0; i < n; i++) {
            off[i] = total_chars;
            len[i] = v[i].size();
            total_chars += len[i];
        }
        vector<char> flat(total_chars);
        for (int i = 0; i < n; i++)
            memcpy(&flat[off[i]], v[i].c_str(), len[i]);

        cudaMalloc(&d_data, total_chars);
        cudaMalloc(&d_off, n * sizeof(int));
        cudaMalloc(&d_len, n * sizeof(int));
        cudaMemcpy(d_data, flat.data(), total_chars, cudaMemcpyHostToDevice);
        cudaMemcpy(d_off, off.data(), n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_len, len.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    }
    void free() {
        cudaFree(d_data); cudaFree(d_off); cudaFree(d_len);
    }
};

// ---- Kernel 1: 单 segment PT — 直接输出 value ----
__global__ void kernel_single(
    const char* vals, const int* offs, const int* lens,
    char* out, int stride, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;

    int o = offs[i], l = lens[i];
    char* dst = out + (size_t)i * stride;
    for (int j = 0; j < l; j++) dst[j] = vals[o + j];
    dst[l] = '\0';
}

// ---- Kernel 2: 多 segment PT — 前缀 + value ----
__global__ void kernel_multi(
    const char* prefix, int plen,
    const char* vals, const int* offs, const int* lens,
    char* out, int stride, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;

    int o = offs[i], l = lens[i];
    char* dst = out + (size_t)i * stride;
    for (int j = 0; j < plen; j++) dst[j] = prefix[j];
    for (int j = 0; j < l; j++) dst[plen + j] = vals[o + j];
    dst[plen + l] = '\0';
}

// ---- GPU 版 Generate：用 CUDA kernel 替代串行 for 循环 ----
void PriorityQueue::Generate_gpu(PT pt)
{
    CalProb(pt);
    int threads = THREADS;
    char *d_out = nullptr, *d_pref = nullptr;

    if (pt.content.size() == 1)
    {
        segment *a;
        if (pt.content[0].type == 1)
            a = &m.letters[m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2)
            a = &m.digits[m.FindDigit(pt.content[0])];
        else
            a = &m.symbols[m.FindSymbol(pt.content[0])];

        int total = pt.max_indices[0];
        if (total == 0) { total_guesses = 0; return; }

        FlatStr fs; fs.alloc(a->ordered_values);
        cudaMalloc(&d_out, (size_t)total * MAX_PW_LEN);
        int blocks = (total + threads - 1) / threads;

        // ★ GPU kernel 调用：取代 CPU 串行 for 循环
        kernel_single<<<blocks, threads>>>(
            fs.d_data, fs.d_off, fs.d_len, d_out, MAX_PW_LEN, total);
        cudaDeviceSynchronize();

        // 将 GPU 结果拷回 Host
        vector<char> h_out((size_t)total * MAX_PW_LEN);
        cudaMemcpy(h_out.data(), d_out,
                   (size_t)total * MAX_PW_LEN, cudaMemcpyDeviceToHost);
        guesses.clear();
        for (int i = 0; i < total; i++)
            guesses.push_back(string(&h_out[i * MAX_PW_LEN]));

        fs.free();
    }
    else
    {
        // 构建前缀（固定部分）
        string prefix;
        int seg_idx = 0;
        for (int idx : pt.curr_indices)
        {
            if (pt.content[seg_idx].type == 1)
                prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            else if (pt.content[seg_idx].type == 2)
                prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            else
                prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            seg_idx++;
            if (seg_idx == (int)pt.content.size() - 1) break;
        }

        segment *a;
        int last = pt.content.size() - 1;
        if (pt.content[last].type == 1)
            a = &m.letters[m.FindLetter(pt.content[last])];
        else if (pt.content[last].type == 2)
            a = &m.digits[m.FindDigit(pt.content[last])];
        else
            a = &m.symbols[m.FindSymbol(pt.content[last])];

        int total = pt.max_indices[last];
        if (total == 0) { total_guesses = 0; return; }

        int plen = prefix.size();
        cudaMalloc(&d_pref, plen + 1);
        cudaMemcpy(d_pref, prefix.c_str(), plen + 1, cudaMemcpyHostToDevice);

        FlatStr fs; fs.alloc(a->ordered_values);
        cudaMalloc(&d_out, (size_t)total * MAX_PW_LEN);
        int blocks = (total + threads - 1) / threads;

        // ★ GPU kernel 调用：取代 CPU 串行 for 循环
        kernel_multi<<<blocks, threads>>>(
            d_pref, plen, fs.d_data, fs.d_off, fs.d_len,
            d_out, MAX_PW_LEN, total);
        cudaDeviceSynchronize();

        vector<char> h_out((size_t)total * MAX_PW_LEN);
        cudaMemcpy(h_out.data(), d_out,
                   (size_t)total * MAX_PW_LEN, cudaMemcpyDeviceToHost);
        guesses.clear();
        for (int i = 0; i < total; i++)
            guesses.push_back(string(&h_out[i * MAX_PW_LEN]));

        fs.free();
    }

    if (d_out)  cudaFree(d_out);
    if (d_pref) cudaFree(d_pref);
    total_guesses = guesses.size();
}

void PriorityQueue::PopNext_gpu()
{
    Generate_gpu(priority.front());

    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        CalProb(pt);
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            if (iter != priority.end()-1 && iter != priority.begin())
            {
                if (pt.prob <= iter->prob && pt.prob > (iter+1)->prob)
                { priority.emplace(iter+1, pt); break; }
            }
            if (iter == priority.end()-1)
            { priority.emplace_back(pt); break; }
            if (iter == priority.begin() && iter->prob < pt.prob)
            { priority.emplace(iter, pt); break; }
        }
    }
    priority.erase(priority.begin());
}

// ---- GPU 初始化 ----
void GPU_Init()
{
    int n;
    cudaGetDeviceCount(&n);
    if (n == 0) { cerr << "[GPU] No GPU found!" << endl; exit(1); }
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    cout << "[GPU] Device: " << p.name
         << " (" << p.multiProcessorCount << " SMs, "
         << p.totalGlobalMem / (1024*1024) << " MB)" << endl;
}
