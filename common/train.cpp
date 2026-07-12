#include "PCFG.h"
#include <fstream>
#include <cctype>
#include <algorithm>

// ============================================================
//  model::train  — 从口令字典训练 PCFG 模型
//
//  并行化思路（可选）：
//    由于统计量可加，可以将训练集拆分到多个线程/进程分别
//    训练，最后将各子模型的 segment/PT 频率累加合并。
//    本文件提供的是串行基线版本。
// ============================================================

void model::train(string path)
{
    string pw;
    ifstream train_set(path);
    int lines = 0;
    cout << "Training..." << endl;
    cout << "Training phase 1: reading and parsing passwords..." << endl;

    while (train_set >> pw)
    {
        lines++;
        if (lines % 100000 == 0)
            cout << "Lines processed: " << lines << endl;
        if (lines > 3000000) break;   // 训练集上限，可按需调整
        parse(pw);
    }
    train_set.close();
}

// ============================================================
//  model::parse  — 将单条口令解析为 PT + segment 序列
//
//  例如: "password123" → L8D3
//        "hello!99"    → L5S1D2
// ============================================================
void model::parse(string pw)
{
    PT pt;
    string curr_part = "";
    int curr_type = 0;   // 0: 未设置, 1: 字母, 2: 数字, 3: 特殊字符

    for (char ch : pw)
    {
        if (isalpha(ch))
        {
            if (curr_type != 1)
            {
                if (curr_type == 2)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindDigit(seg) == -1)
                    {
                        int id = GetNextDigitsID();
                        digits.emplace_back(seg);
                        digits[id].insert(curr_part);
                        digits_freq[id] = 1;
                    }
                    else
                    {
                        int id = FindDigit(seg);
                        digits_freq[id] += 1;
                        digits[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
                else if (curr_type == 3)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindSymbol(seg) == -1)
                    {
                        int id = GetNextSymbolsID();
                        symbols.emplace_back(seg);
                        symbols_freq[id] = 1;
                        symbols[id].insert(curr_part);
                    }
                    else
                    {
                        int id = FindSymbol(seg);
                        symbols_freq[id] += 1;
                        symbols[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
            }
            curr_type = 1;
            curr_part += ch;
        }
        else if (isdigit(ch))
        {
            if (curr_type != 2)
            {
                if (curr_type == 1)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindLetter(seg) == -1)
                    {
                        int id = GetNextLettersID();
                        letters.emplace_back(seg);
                        letters_freq[id] = 1;
                        letters[id].insert(curr_part);
                    }
                    else
                    {
                        int id = FindLetter(seg);
                        letters_freq[id] += 1;
                        letters[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
                else if (curr_type == 3)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindSymbol(seg) == -1)
                    {
                        int id = GetNextSymbolsID();
                        symbols.emplace_back(seg);
                        symbols_freq[id] = 1;
                        symbols[id].insert(curr_part);
                    }
                    else
                    {
                        int id = FindSymbol(seg);
                        symbols_freq[id] += 1;
                        symbols[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
            }
            curr_type = 2;
            curr_part += ch;
        }
        else
        {
            if (curr_type != 3)
            {
                if (curr_type == 1)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindLetter(seg) == -1)
                    {
                        int id = GetNextLettersID();
                        letters.emplace_back(seg);
                        letters_freq[id] = 1;
                        letters[id].insert(curr_part);
                    }
                    else
                    {
                        int id = FindLetter(seg);
                        letters_freq[id] += 1;
                        letters[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
                else if (curr_type == 2)
                {
                    segment seg(curr_type, curr_part.length());
                    if (FindDigit(seg) == -1)
                    {
                        int id = GetNextDigitsID();
                        digits.emplace_back(seg);
                        digits_freq[id] = 1;
                        digits[id].insert(curr_part);
                    }
                    else
                    {
                        int id = FindDigit(seg);
                        digits_freq[id] += 1;
                        digits[id].insert(curr_part);
                    }
                    curr_part.clear();
                    pt.insert(seg);
                }
            }
            curr_type = 3;
            curr_part += ch;
        }
    }

    // 处理最后一个 segment
    if (!curr_part.empty())
    {
        if (curr_type == 1)
        {
            segment seg(curr_type, curr_part.length());
            if (FindLetter(seg) == -1)
            {
                int id = GetNextLettersID();
                letters.emplace_back(seg);
                letters_freq[id] = 1;
                letters[id].insert(curr_part);
            }
            else
            {
                int id = FindLetter(seg);
                letters_freq[id] += 1;
                letters[id].insert(curr_part);
            }
        }
        else if (curr_type == 2)
        {
            segment seg(curr_type, curr_part.length());
            if (FindDigit(seg) == -1)
            {
                int id = GetNextDigitsID();
                digits.emplace_back(seg);
                digits_freq[id] = 1;
                digits[id].insert(curr_part);
            }
            else
            {
                int id = FindDigit(seg);
                digits_freq[id] += 1;
                digits[id].insert(curr_part);
            }
        }
        else
        {
            segment seg(curr_type, curr_part.length());
            if (FindSymbol(seg) == -1)
            {
                int id = GetNextSymbolsID();
                symbols.emplace_back(seg);
                symbols_freq[id] = 1;
                symbols[id].insert(curr_part);
            }
            else
            {
                int id = FindSymbol(seg);
                symbols_freq[id] += 1;
                symbols[id].insert(curr_part);
            }
        }
        curr_part.clear();
        pt.insert(seg);
    }

    total_preterm += 1;
    if (FindPT(pt) == -1)
    {
        for (size_t i = 0; i < pt.content.size(); i++)
            pt.curr_indices.emplace_back(0);
        int id = GetNextPretermID();
        preterminals.emplace_back(pt);
        preterm_freq[id] = 1;
    }
    else
    {
        int id = FindPT(pt);
        preterm_freq[id] += 1;
    }
}

// ============================================================
//  查找函数
// ============================================================
int model::FindPT(PT pt)
{
    for (int id = 0; id < (int)preterminals.size(); id++)
    {
        if (preterminals[id].content.size() != pt.content.size()) continue;
        bool equal_flag = true;
        for (size_t idx = 0; idx < preterminals[id].content.size(); idx++)
        {
            if (preterminals[id].content[idx].type   != pt.content[idx].type ||
                preterminals[id].content[idx].length != pt.content[idx].length)
            { equal_flag = false; break; }
        }
        if (equal_flag) return id;
    }
    return -1;
}

int model::FindLetter(segment seg)
{
    for (int id = 0; id < (int)letters.size(); id++)
        if (letters[id].length == seg.length) return id;
    return -1;
}

int model::FindDigit(segment seg)
{
    for (int id = 0; id < (int)digits.size(); id++)
        if (digits[id].length == seg.length) return id;
    return -1;
}

int model::FindSymbol(segment seg)
{
    for (int id = 0; id < (int)symbols.size(); id++)
        if (symbols[id].length == seg.length) return id;
    return -1;
}

// ============================================================
//  segment 成员函数
// ============================================================
void PT::insert(segment seg) { content.emplace_back(seg); }

void segment::insert(string value)
{
    if (values.find(value) == values.end())
    {
        values[value] = values.size();
        freqs[values[value]] = 1;
    }
    else
    {
        freqs[values[value]] += 1;
    }
}

void segment::order()
{
    // 将 values 按频率降序存入 ordered_values
    for (pair<string, int> value : values)
        ordered_values.emplace_back(value.first);

    sort(ordered_values.begin(), ordered_values.end(),
         [this](const string &a, const string &b)
         { return freqs.at(values[a]) > freqs.at(values[b]); });

    // 计算 ordered_freqs 和 total_freq（修复：原先重复累加导致双倍计数）
    ordered_freqs.clear();
    total_freq = 0;
    for (const string &val : ordered_values)
    {
        int f = freqs.at(values[val]);
        ordered_freqs.emplace_back(f);
        total_freq += f;
    }
}

void segment::PrintSeg()
{
    if (type == 1) cout << "L" << length;
    if (type == 2) cout << "D" << length;
    if (type == 3) cout << "S" << length;
}

void segment::PrintValues()
{
    for (string iter : ordered_values)
        cout << iter << " freq:" << freqs[values[iter]] << endl;
}

void PT::PrintPT()
{
    for (auto iter : content) iter.PrintSeg();
}

void model::print()
{
    cout << "preterminals:" << endl;
    for (int i = 0; i < (int)preterminals.size(); i++)
    {
        preterminals[i].PrintPT();
        cout << " freq:" << preterm_freq[i] << endl;
    }
    for (auto iter : ordered_pts)
    {
        iter.PrintPT();
        cout << " freq:" << preterm_freq[FindPT(iter)] << endl;
    }
    cout << "segments:" << endl;
    for (int i = 0; i < (int)letters.size(); i++)
    {
        letters[i].PrintSeg();
        cout << " freq:" << letters_freq[i] << endl;
    }
    for (int i = 0; i < (int)digits.size(); i++)
    {
        digits[i].PrintSeg();
        cout << " freq:" << digits_freq[i] << endl;
    }
    for (int i = 0; i < (int)symbols.size(); i++)
    {
        symbols[i].PrintSeg();
        cout << " freq:" << symbols_freq[i] << endl;
    }
}

void model::order()
{
    cout << "Training phase 2: Ordering segment values and PTs..." << endl;
    for (PT pt : preterminals)
    {
        pt.preterm_prob = float(preterm_freq[FindPT(pt)]) / total_preterm;
        ordered_pts.emplace_back(pt);
    }
    cout << "total pts: " << ordered_pts.size() << endl;
    sort(ordered_pts.begin(), ordered_pts.end(),
         [](const PT &a, const PT &b) { return a.preterm_prob > b.preterm_prob; });

    cout << "Ordering letters..." << endl;
    for (int i = 0; i < (int)letters.size(); i++) letters[i].order();
    cout << "Ordering digits..."  << endl;
    for (int i = 0; i < (int)digits.size();  i++) digits[i].order();
    cout << "Ordering symbols..." << endl;
    for (int i = 0; i < (int)symbols.size(); i++) symbols[i].order();
}

void model::store(string store_path) { /* 模型持久化 — 可选实现 */ }
void model::load(string load_path)    { /* 模型加载   — 可选实现 */ }
