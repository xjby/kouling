#include "../common/PCFG.h"
#include "../common/md5.h"
#include <omp.h>

// ============================================================
//  OpenMP 版本 — 多线程并行口令猜测生成
//
//  并行化策略：
//    - Generate() 中的 for 循环用 #pragma omp parallel for 并行化
//    - 线程间共享 segment 数据（只读），各自写入 guesses
//    - 使用 critical 段保护 guesses 的并发写入
//    - 更优方案：每线程分配独立的本地 buffer，最后合并
//
//  关键指令：
//    #pragma omp parallel for schedule(dynamic)
//      — 动态调度，适应各 segment value 数目不均的情况
//    #pragma omp critical
//      — 保护 vector<string> 的并发 push_back
// ============================================================

void PriorityQueue::CalProb(PT &pt)
{
    pt.prob = pt.preterm_prob;
    int index = 0;

    for (int idx : pt.curr_indices)
    {
        if (pt.content[index].type == 1)
        {
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 2)
        {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 3)
        {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
        }
        index++;
    }
}

void PriorityQueue::init()
{
    for (PT pt : m.ordered_pts)
    {
        for (segment seg : pt.content)
        {
            if (seg.type == 1)
                pt.max_indices.emplace_back(
                    m.letters[m.FindLetter(seg)].ordered_values.size());
            if (seg.type == 2)
                pt.max_indices.emplace_back(
                    m.digits[m.FindDigit(seg)].ordered_values.size());
            if (seg.type == 3)
                pt.max_indices.emplace_back(
                    m.symbols[m.FindSymbol(seg)].ordered_values.size());
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        CalProb(pt);
        priority.emplace_back(pt);
    }
}

void PriorityQueue::Generate(PT pt)
{
    CalProb(pt);

    if (pt.content.size() == 1)
    {
        // ============================================================
        //  单 segment PT：OpenMP 并行遍历所有 value
        //
        // ★ OpenMP 核心优化点 1
        //   schedule(dynamic): 当各线程工作量不均时，动态分配迭代
        //   这里每个 value 直接就是一个猜测，工作量均匀，用 static 即可
        // ============================================================
        segment *a;
        if (pt.content[0].type == 1)
            a = &m.letters[m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2)
            a = &m.digits[m.FindDigit(pt.content[0])];
        else
            a = &m.symbols[m.FindSymbol(pt.content[0])];

        int total = pt.max_indices[0];

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < total; i++)
        {
            string guess = a->ordered_values[i];
            #pragma omp critical
            {
                guesses.emplace_back(guess);
                total_guesses++;
            }
        }
    }
    else
    {
        // ============================================================
        //  多 segment PT：先固定前缀，再 OpenMP 并行遍历最后一个
        //
        // ★ OpenMP 核心优化点 2
        //   前缀在并行区外提前计算好（只读共享），
        //   并行区内仅做字符串拼接和 guesses 写入。
        // ============================================================
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

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < total; i++)
        {
            string guess = prefix + a->ordered_values[i];
            #pragma omp critical
            {
                guesses.emplace_back(guess);
                total_guesses++;
            }
        }
    }
}

void PriorityQueue::PopNext()
{
    Generate(priority.front());

    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts)
    {
        CalProb(pt);
        for (auto iter = priority.begin(); iter != priority.end(); iter++)
        {
            if (iter != priority.end() - 1 && iter != priority.begin())
            {
                if (pt.prob <= iter->prob && pt.prob > (iter + 1)->prob)
                { priority.emplace(iter + 1, pt); break; }
            }
            if (iter == priority.end() - 1)
            { priority.emplace_back(pt); break; }
            if (iter == priority.begin() && iter->prob < pt.prob)
            { priority.emplace(iter, pt); break; }
        }
    }
    priority.erase(priority.begin());
}

vector<PT> PT::NewPTs()
{
    vector<PT> res;
    if (content.size() == 1) return res;

    int init_pivot = pivot;
    for (int i = pivot; i < (int)curr_indices.size() - 1; i++)
    {
        curr_indices[i]++;
        if (curr_indices[i] < max_indices[i])
        { pivot = i; res.emplace_back(*this); }
        curr_indices[i]--;
    }
    pivot = init_pivot;
    return res;
}
