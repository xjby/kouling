#ifndef PCFG_H
#define PCFG_H

#include <string>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <algorithm>

using namespace std;

// ============================================================
//  PCFG (Probabilistic Context-Free Grammar) 口令猜测模型
//  公共头文件 — 定义所有并行版本共用的基础数据结构
// ============================================================

class segment
{
public:
    int type;    // 0: 未设置, 1: 字母(L), 2: 数字(D), 3: 特殊字符(S)
    int length;  // 段长度, e.g. S6 长度为 6
    segment(int type, int length) : type(type), length(length) {}

    void PrintSeg();
    void insert(string value);
    void order();
    void PrintValues();

    vector<string> ordered_values;       // 按概率降序排列的 value 列表
    vector<int>    ordered_freqs;        // 对应频率（分子）
    int            total_freq = 0;       // 总频率（分母）
    unordered_map<string, int> values;   // value → id 映射
    unordered_map<int, int>    freqs;    // id → 频数
};

class PT   // Pre-Terminal: 由若干 segment 组成的模板
{
public:
    vector<segment> content;       // e.g. L6D1 → [L6, D1]
    int pivot = 0;

    void insert(segment seg);
    void PrintPT();
    vector<PT> NewPTs();           // 生成新的 PT 衍生体

    vector<int> curr_indices;      // 各 segment 当前 value 下标
    vector<int> max_indices;       // 各 segment 最大 value 数目
    float preterm_prob;            // PT 本身的概率
    float prob;                    // 当前实例化后的完整概率
};

class model
{
public:
    // ---- ID 分配器 ----
    int preterm_id = -1, letters_id = -1, digits_id = -1, symbols_id = -1;
    int GetNextPretermID() { return ++preterm_id; }
    int GetNextLettersID() { return ++letters_id; }
    int GetNextDigitsID()  { return ++digits_id;  }
    int GetNextSymbolsID() { return ++symbols_id; }

    // ---- 统计数据 ----
    int total_preterm = 0;
    vector<PT> preterminals;
    vector<segment> letters;
    vector<segment> digits;
    vector<segment> symbols;

    unordered_map<int, int> preterm_freq;
    unordered_map<int, int> letters_freq;
    unordered_map<int, int> digits_freq;
    unordered_map<int, int> symbols_freq;

    vector<PT> ordered_pts;   // 按概率降序排列的 PT

    // ---- 查找函数 ----
    int FindPT(PT pt);
    int FindLetter(segment seg);
    int FindDigit(segment seg);
    int FindSymbol(segment seg);

    // ---- 训练 ----
    void train(string train_path);
    void store(string store_path);
    void load(string load_path);
    void parse(string pw);
    void order();
    void print();
};

class PriorityQueue
{
public:
    vector<PT> priority;       // 按概率降序的优先队列（vector 模拟）
    model m;                   // 统计模型
    int total_guesses = 0;
    vector<string> guesses;    // 当前批次生成的猜测

    void CalProb(PT &pt);      // 计算一个 PT 实例化后的概率
    void init();               // 初始化优先队列
    void Generate(PT pt);      // 生成猜测（各并行版本各自实现）
    void PopNext();            // 弹出概率最高的 PT 并处理
};

#endif
