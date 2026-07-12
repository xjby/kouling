#include "../common/PCFG.h"
#include "../common/md5.h"

// ============================================================
//  Serial 版本 — 串行口令猜测生成（基线参考实现）
//
//  本文件实现了 PCFG 优先队列的三个核心操作：
//    1. CalProb   — 计算 PT 实例化后的概率
//    2. init      — 初始化优先队列
//    3. Generate  — 从队首 PT 生成一批口令猜测
//    4. PopNext   — 弹出最高概率 PT，生成猜测，衍生新 PT
//    5. NewPTs    — 从当前 PT 生成新 PT（更改一个 segment 的值）
//
//  所有操作均为串行，作为并行版本的性能对比基线。
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

// ============================================================
//  Generate — 从指定 PT 生成所有可能的猜测（串行版本）
//
//  【并行化热点】以下带 ★ 标记的 for 循环是并行化的主要目标：
//    - 多线程: 用 #pragma omp parallel for 分割迭代
//    - GPU:    将循环映射为 CUDA kernel 线程
//    - 集群:   按 segment value 范围分发给各 MPI 进程
// ============================================================
void PriorityQueue::Generate(PT pt)
{
    CalProb(pt);

    if (pt.content.size() == 1)
    {
        // 只有一个 segment：直接枚举所有 value
        segment *a;
        if (pt.content[0].type == 1)
            a = &m.letters[m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2)
            a = &m.digits[m.FindDigit(pt.content[0])];
        else
            a = &m.symbols[m.FindSymbol(pt.content[0])];

        // ★ 并行化热点 1：遍历 segment 所有 value
        for (int i = 0; i < pt.max_indices[0]; i++)
        {
            string guess = a->ordered_values[i];
            guesses.emplace_back(guess);
            total_guesses++;
        }
    }
    else
    {
        // 多个 segment：固定前 N-1 个，枚举最后一个
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

        // ★ 并行化热点 2：遍历最后一个 segment 的所有 value
        for (int i = 0; i < pt.max_indices[last]; i++)
        {
            string guess = prefix + a->ordered_values[i];
            guesses.emplace_back(guess);
            total_guesses++;
        }
    }
}

void PriorityQueue::PopNext()
{
    Generate(priority.front());

    // 生成衍生 PT 并插入优先队列
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
