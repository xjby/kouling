#include "PCFG.h"
#include <vector>
#include <cstring>
#include <iostream>
#include <cuda_runtime.h>
using namespace std;

#define MAX_PW_LEN 128
#define THREADS 256

struct FlatStr {
    char *d_data; int *d_off, *d_len; int total_chars, n;
    void alloc(const vector<string>& v) {
        n = v.size();
        vector<int> off(n), len(n);
        total_chars = 0;
        for (int i = 0; i < n; i++) { off[i] = total_chars; len[i] = v[i].size(); total_chars += len[i]; }
        vector<char> flat(total_chars);
        for (int i = 0; i < n; i++) memcpy(&flat[off[i]], v[i].c_str(), len[i]);
        cudaMalloc(&d_data, total_chars);
        cudaMalloc(&d_off, n * sizeof(int));
        cudaMalloc(&d_len, n * sizeof(int));
        cudaMemcpy(d_data, flat.data(), total_chars, cudaMemcpyHostToDevice);
        cudaMemcpy(d_off, off.data(), n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_len, len.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    }
    void free() { cudaFree(d_data); cudaFree(d_off); cudaFree(d_len); }
};

__global__ void kernel_single(const char* vals, const int* offs, const int* lens,
                               char* out, int stride, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int o = offs[i], l = lens[i];
    char* dst = out + (size_t)i * stride;
    for (int j = 0; j < l; j++) dst[j] = vals[o + j];
    dst[l] = '\0';
}

__global__ void kernel_multi(const char* prefix, int plen,
                              const char* vals, const int* offs, const int* lens,
                              char* out, int stride, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int o = offs[i], l = lens[i];
    char* dst = out + (size_t)i * stride;
    for (int j = 0; j < plen; j++) dst[j] = prefix[j];
    for (int j = 0; j < l; j++) dst[plen + j] = vals[o + j];
    dst[plen + l] = '\0';
}

void PriorityQueue::Generate_gpu(PT pt) {
    CalProb(pt);
    int threads = THREADS;
    char *d_out = nullptr, *d_pref = nullptr;

    if (pt.content.size() == 1) {
        segment *a;
        if (pt.content[0].type == 1) a = &m.letters[m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2) a = &m.digits[m.FindDigit(pt.content[0])];
        else a = &m.symbols[m.FindSymbol(pt.content[0])];
        int total = pt.max_indices[0];
        if (total == 0) { total_guesses = 0; return; }
        FlatStr fs; fs.alloc(a->ordered_values);
        cudaMalloc(&d_out, (size_t)total * MAX_PW_LEN);
        int blocks = (total + threads - 1) / threads;
        kernel_single<<<blocks, threads>>>(fs.d_data, fs.d_off, fs.d_len, d_out, MAX_PW_LEN, total);
        cudaDeviceSynchronize();
        vector<char> h_out((size_t)total * MAX_PW_LEN);
        cudaMemcpy(h_out.data(), d_out, (size_t)total * MAX_PW_LEN, cudaMemcpyDeviceToHost);
        guesses.clear();
        for (int i = 0; i < total; i++) guesses.push_back(string(&h_out[i * MAX_PW_LEN]));
        fs.free();
    } else {
        string prefix;
        int seg_idx = 0;
        for (int idx : pt.curr_indices) {
            if (pt.content[seg_idx].type == 1)
                prefix += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            else if (pt.content[seg_idx].type == 2)
                prefix += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            else
                prefix += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            seg_idx++;
            if (seg_idx == pt.content.size() - 1) break;
        }
        segment *a;
        int last = pt.content.size() - 1;
        if (pt.content[last].type == 1) a = &m.letters[m.FindLetter(pt.content[last])];
        else if (pt.content[last].type == 2) a = &m.digits[m.FindDigit(pt.content[last])];
        else a = &m.symbols[m.FindSymbol(pt.content[last])];
        int total = pt.max_indices[last];
        if (total == 0) { total_guesses = 0; return; }
        int plen = prefix.size();
        cudaMalloc(&d_pref, plen + 1);
        cudaMemcpy(d_pref, prefix.c_str(), plen + 1, cudaMemcpyHostToDevice);
        FlatStr fs; fs.alloc(a->ordered_values);
        cudaMalloc(&d_out, (size_t)total * MAX_PW_LEN);
        int blocks = (total + threads - 1) / threads;
        kernel_multi<<<blocks, threads>>>(d_pref, plen, fs.d_data, fs.d_off, fs.d_len, d_out, MAX_PW_LEN, total);
        cudaDeviceSynchronize();
        vector<char> h_out((size_t)total * MAX_PW_LEN);
        cudaMemcpy(h_out.data(), d_out, (size_t)total * MAX_PW_LEN, cudaMemcpyDeviceToHost);
        guesses.clear();
        for (int i = 0; i < total; i++) guesses.push_back(string(&h_out[i * MAX_PW_LEN]));
        fs.free();
    }
    if (d_out) cudaFree(d_out);
    if (d_pref) cudaFree(d_pref);
    total_guesses = guesses.size();
}

void GPU_Init() {
    int n; cudaGetDeviceCount(&n);
    if (n == 0) { cerr << "No GPU found!" << endl; exit(1); }
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    cout << "GPU: " << p.name << endl;
}
