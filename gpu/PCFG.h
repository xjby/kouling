#ifndef PCFG_GPU_H
#define PCFG_GPU_H

#include <string>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cstring>

using namespace std;

class segment
{
public:
    int type, length;
    segment(int type, int length) : type(type), length(length) {}
    void PrintSeg();
    void insert(string value);
    void order();
    void PrintValues();
    vector<string> ordered_values;
    vector<int> ordered_freqs;
    int total_freq = 0;
    unordered_map<string, int> values;
    unordered_map<int, int> freqs;
};

class PT
{
public:
    vector<segment> content;
    int pivot = 0;
    void insert(segment seg);
    void PrintPT();
    vector<PT> NewPTs();
    vector<int> curr_indices;
    vector<int> max_indices;
    float preterm_prob;
    float prob;
};

class model
{
public:
    int preterm_id = -1, letters_id = -1, digits_id = -1, symbols_id = -1;
    int GetNextPretermID() { return ++preterm_id; }
    int GetNextLettersID() { return ++letters_id; }
    int GetNextDigitsID()  { return ++digits_id;  }
    int GetNextSymbolsID() { return ++symbols_id; }
    int total_preterm = 0;
    vector<PT> preterminals;
    vector<segment> letters, digits, symbols;
    unordered_map<int, int> preterm_freq, letters_freq, digits_freq, symbols_freq;
    vector<PT> ordered_pts;
    int FindPT(PT pt);
    int FindLetter(segment seg);
    int FindDigit(segment seg);
    int FindSymbol(segment seg);
    void train(string train_path);
    void parse(string pw);
    void order();
    void print();
};

class PriorityQueue
{
public:
    vector<PT> priority;
    model m;
    void CalProb(PT &pt);
    void init();
    void Generate_gpu(PT pt);    // GPU 版本口令生成
    void PopNext_gpu();
    int total_guesses = 0;
    vector<string> guesses;
};

// GPU 初始化
void GPU_Init();

#endif
