#ifndef PCFG_H
#define PCFG_H

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
    int type;
    int length;
    segment(int type, int length)
    {
        this->type = type;
        this->length = length;
    };

    void PrintSeg();
    vector<string> ordered_values;
    vector<int> ordered_freqs;
    int total_freq = 0;
    unordered_map<string, int> values;
    unordered_map<int, int> freqs;

    void insert(string value);
    void order();
    void PrintValues();
}; // <--- 这里补上了分号

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
}; // <--- 这里补上了分号

class model
{
public:
    int preterm_id = -1;
    int letters_id = -1;
    int digits_id = -1;
    int symbols_id = -1;
    
    int GetNextPretermID() { return ++preterm_id; }
    int GetNextLettersID() { return ++letters_id; }
    int GetNextDigitsID() { return ++digits_id; }
    int GetNextSymbolsID() { return ++symbols_id; }

    int total_preterm = 0;
    vector<PT> preterminals;
    int FindPT(PT pt);

    vector<segment> letters;
    vector<segment> digits;
    vector<segment> symbols;
    int FindLetter(segment seg);
    int FindDigit(segment seg);
    int FindSymbol(segment seg);

    unordered_map<int, int> preterm_freq;
    unordered_map<int, int> letters_freq;
    unordered_map<int, int> digits_freq;
    unordered_map<int, int> symbols_freq;

    vector<PT> ordered_pts;

    void train(string train_path);
    void store(string store_path);
    void load(string load_path);
    void parse(string pw);
    void order();
    void print();
}; // <--- 这里补上了分号

class PriorityQueue
{
public:
    vector<PT> priority;
    model m;
    void Generate_mpi(PT pt);
    void Generate_gpu(PT pt);     // GPU 版本的口令生成
    void HashBatch_gpu();         // GPU 版本的批量 MD5
    void CalProb(PT &pt);
    void init();
    void Generate(PT pt);
    void Generate_cpu_serial(PT pt);  // CPU 串行版本（用于性能对比）
    void PopNext();
    int total_guesses = 0;
    vector<string> guesses;
}; // <--- 这里补上了分号

#endif