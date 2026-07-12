#include "PCFG.h"
#include <fstream>
#include <cctype>
#include <algorithm>

// ============================================================
//  GPU 版本模型训练
//  与 common/train.cpp 逻辑相同，适配 GPU 版 PCFG.h
// ============================================================

void model::train(string path)
{
    string pw;
    ifstream train_set(path);
    int lines = 0;
    cout << "[GPU] Training..." << endl;
    while (train_set >> pw)
    {
        lines++;
        if (lines % 100000 == 0)
            cout << "[GPU] Lines processed: " << lines << endl;
        if (lines > 3000000) break;
        parse(pw);
    }
    train_set.close();
}

void model::parse(string pw)
{
    PT pt;
    string curr_part = "";
    int curr_type = 0;

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

    if (!curr_part.empty())
    {
        if (curr_type == 1)
        {
            segment seg(curr_type, curr_part.length());
            if (FindLetter(seg) == -1)
            { int id = GetNextLettersID(); letters.emplace_back(seg); letters_freq[id] = 1; letters[id].insert(curr_part); }
            else
            { int id = FindLetter(seg); letters_freq[id] += 1; letters[id].insert(curr_part); }
        }
        else if (curr_type == 2)
        {
            segment seg(curr_type, curr_part.length());
            if (FindDigit(seg) == -1)
            { int id = GetNextDigitsID(); digits.emplace_back(seg); digits_freq[id] = 1; digits[id].insert(curr_part); }
            else
            { int id = FindDigit(seg); digits_freq[id] += 1; digits[id].insert(curr_part); }
        }
        else
        {
            segment seg(curr_type, curr_part.length());
            if (FindSymbol(seg) == -1)
            { int id = GetNextSymbolsID(); symbols.emplace_back(seg); symbols_freq[id] = 1; symbols[id].insert(curr_part); }
            else
            { int id = FindSymbol(seg); symbols_freq[id] += 1; symbols[id].insert(curr_part); }
        }
        curr_part.clear();
        pt.insert(seg);
    }

    total_preterm += 1;
    if (FindPT(pt) == -1)
    {
        for (size_t i = 0; i < pt.content.size(); i++) pt.curr_indices.emplace_back(0);
        int id = GetNextPretermID();
        preterminals.emplace_back(pt);
        preterm_freq[id] = 1;
    }
    else
    {
        preterm_freq[FindPT(pt)] += 1;
    }
}

int model::FindPT(PT pt) {
    for (int id = 0; id < (int)preterminals.size(); id++) {
        if (preterminals[id].content.size() != pt.content.size()) continue;
        bool eq = true;
        for (size_t i = 0; i < preterminals[id].content.size(); i++)
            if (preterminals[id].content[i].type != pt.content[i].type ||
                preterminals[id].content[i].length != pt.content[i].length)
            { eq = false; break; }
        if (eq) return id;
    }
    return -1;
}

int model::FindLetter(segment seg) {
    for (int i = 0; i < (int)letters.size(); i++)
        if (letters[i].length == seg.length) return i;
    return -1;
}
int model::FindDigit(segment seg) {
    for (int i = 0; i < (int)digits.size(); i++)
        if (digits[i].length == seg.length) return i;
    return -1;
}
int model::FindSymbol(segment seg) {
    for (int i = 0; i < (int)symbols.size(); i++)
        if (symbols[i].length == seg.length) return i;
    return -1;
}

void PT::insert(segment seg) { content.emplace_back(seg); }
void segment::insert(string value) {
    if (values.find(value) == values.end()) { values[value] = values.size(); freqs[values[value]] = 1; }
    else { freqs[values[value]] += 1; }
}
void segment::order() {
    for (auto& v : values) ordered_values.emplace_back(v.first);
    sort(ordered_values.begin(), ordered_values.end(),
         [this](const string &a, const string &b) { return freqs[values[a]] > freqs[values[b]]; });
    ordered_freqs.clear(); total_freq = 0;
    for (const string &val : ordered_values) { int f = freqs[values[val]]; ordered_freqs.emplace_back(f); total_freq += f; }
}
void segment::PrintSeg() {
    if (type == 1) cout << "L" << length;
    if (type == 2) cout << "D" << length;
    if (type == 3) cout << "S" << length;
}
void PT::PrintPT() { for (auto& s : content) s.PrintSeg(); }

void model::order() {
    cout << "[GPU] Ordering..." << endl;
    for (PT pt : preterminals) {
        pt.preterm_prob = float(preterm_freq[FindPT(pt)]) / total_preterm;
        ordered_pts.emplace_back(pt);
    }
    sort(ordered_pts.begin(), ordered_pts.end(),
         [](const PT &a, const PT &b) { return a.preterm_prob > b.preterm_prob; });
    for (int i = 0; i < (int)letters.size(); i++) letters[i].order();
    for (int i = 0; i < (int)digits.size(); i++) digits[i].order();
    for (int i = 0; i < (int)symbols.size(); i++) symbols[i].order();
}
