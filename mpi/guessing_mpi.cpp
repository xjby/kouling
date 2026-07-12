#include "../common/PCFG.h"
#include "../common/md5.h"
#include <mpi.h>

// ============================================================
//  MPI 版本 — 分布式口令猜测生成
//
//  并行化策略：
//    方案：主从 (Master-Worker) 模型
//      - Rank 0：维护优先队列，将 Generate 任务分发给各 worker
//      - Rank 1..N-1：接收 PT 参数，在自己的 segment 值子集上
//        生成猜测，计算 MD5，结果回传 rank 0
//
//    更简单的方案（本实现采用）：
//      - 所有 rank 加载相同的模型
//      - Rank 0 运行优先队列主逻辑
//      - 当需要 Generate 时，将 last segment 的 value 范围
//        按 rank 数量平均分配，各 rank 只生成自己的子集
//      - 最后用 MPI_Gatherv 汇总所有 rank 的猜测
//
//  通信模式：
//    - MPI_Bcast:  广播 PT 参数（前缀、segment 类型等）
//    - MPI_Scatter: 分发 value 子集范围
//    - MPI_Gather:  收集各 rank 生成的猜测数量
//    - MPI_Gatherv: 收集各 rank 的实际猜测字符串
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
//  Generate_mpi — MPI 分布式猜测生成
//
//  将最后一个 segment 的 value 索引范围 [0, total) 按 rank 数
//  均匀分块。每个 rank 只生成自己分块内的猜测，
//  然后通过 MPI_Gatherv 汇总到 rank 0。
//
//  @param pt      当前 PT
//  @param rank    当前进程在 MPI_COMM_WORLD 中的 rank
//  @param nprocs  MPI_COMM_WORLD 中的进程总数
// ============================================================
void PriorityQueue::Generate_mpi(PT pt, int rank, int nprocs)
{
    CalProb(pt);

    int total = 0;
    segment *a = nullptr;
    string prefix = "";

    if (pt.content.size() == 1)
    {
        if (pt.content[0].type == 1)
            a = &m.letters[m.FindLetter(pt.content[0])];
        else if (pt.content[0].type == 2)
            a = &m.digits[m.FindDigit(pt.content[0])];
        else
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        total = pt.max_indices[0];
    }
    else
    {
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

        int last = pt.content.size() - 1;
        if (pt.content[last].type == 1)
            a = &m.letters[m.FindLetter(pt.content[last])];
        else if (pt.content[last].type == 2)
            a = &m.digits[m.FindDigit(pt.content[last])];
        else
            a = &m.symbols[m.FindSymbol(pt.content[last])];
        total = pt.max_indices[last];
    }

    if (total == 0) return;

    // ---- 按 rank 均分 value 索引范围 ----
    int chunk     = total / nprocs;
    int remainder = total % nprocs;
    int my_start  = rank * chunk + (rank < remainder ? rank : remainder);
    int my_end    = my_start + chunk + (rank < remainder ? 1 : 0);

    // 各 rank 生成自己分块内的猜测
    vector<string> local_guesses;
    for (int i = my_start; i < my_end; i++)
    {
        local_guesses.emplace_back(prefix + a->ordered_values[i]);
    }

    int local_count = local_guesses.size();

    // ---- 第一步：收集各 rank 的猜测数量 ----
    int *counts = nullptr;
    if (rank == 0) counts = new int[nprocs];
    MPI_Gather(&local_count, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // ---- 第二步：计算偏移并收集所有猜测字符串 ----
    int *displs = nullptr;
    int total_global = 0;
    if (rank == 0)
    {
        displs = new int[nprocs];
        displs[0] = 0;
        for (int i = 1; i < nprocs; i++)
            displs[i] = displs[i-1] + counts[i-1];
    }

    // 将所有本地猜测的字符串长度收集到 rank 0
    // （简化处理：假设每条口令最长 256 字符）
    int MAX_LEN = 256;
    char *local_buf = new char[local_count * MAX_LEN];
    for (int i = 0; i < local_count; i++)
    {
        memset(local_buf + i * MAX_LEN, 0, MAX_LEN);
        strncpy(local_buf + i * MAX_LEN, local_guesses[i].c_str(), MAX_LEN - 1);
    }

    char *global_buf = nullptr;
    if (rank == 0)
    {
        for (int i = 0; i < nprocs; i++)
            total_global += counts[i];
        global_buf = new char[total_global * MAX_LEN];
    }

    // 计算每个 rank 的发送量（字节）
    int *send_counts_bytes = nullptr;
    int *recv_counts_bytes = nullptr;
    int *byte_displs = nullptr;
    if (rank == 0)
    {
        send_counts_bytes = new int[nprocs];
        recv_counts_bytes = new int[nprocs];
        byte_displs = new int[nprocs];
        for (int i = 0; i < nprocs; i++)
        {
            recv_counts_bytes[i] = counts[i] * MAX_LEN;
            byte_displs[i] = displs[i] * MAX_LEN;
        }
    }
    int local_bytes = local_count * MAX_LEN;

    MPI_Gatherv(local_buf, local_bytes, MPI_BYTE,
                global_buf, recv_counts_bytes, byte_displs, MPI_BYTE,
                0, MPI_COMM_WORLD);

    // Rank 0 汇总结果到 guesses
    if (rank == 0)
    {
        for (int i = 0; i < total_global; i++)
        {
            string s(global_buf + i * MAX_LEN);
            if (!s.empty()) guesses.emplace_back(s);
        }
        total_guesses = guesses.size();
        delete[] global_buf;
        delete[] counts;
        delete[] displs;
        delete[] send_counts_bytes;
        delete[] recv_counts_bytes;
        delete[] byte_displs;
    }

    delete[] local_buf;
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
