#include "PCFG.h"
#include <chrono>
#include <queue>
#include <cstring>
#include <unistd.h>
#include <sstream>
#include <iostream>
using namespace std;

extern void GPU_Init();

void PriorityQueue::CalProb(PT &pt) {
    pt.prob = pt.preterm_prob;
    int index = 0;
    for (int idx : pt.curr_indices) {
        if (pt.content[index].type == 1) {
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 2) {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 3) {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
        }
        index++;
    }
}

void PriorityQueue::init() {
    for (PT pt : m.ordered_pts) {
        for (segment seg : pt.content) {
            if (seg.type == 1) pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            if (seg.type == 2) pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            if (seg.type == 3) pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        CalProb(pt);
        priority.emplace_back(pt);
    }
}

void PriorityQueue::PopNext() {
    Generate_gpu(priority.front());
    vector<PT> new_pts = priority.front().NewPTs();
    for (PT pt : new_pts) {
        CalProb(pt);
        for (auto iter = priority.begin(); iter != priority.end(); iter++) {
            if (iter != priority.end()-1 && iter != priority.begin()) {
                if (pt.prob <= iter->prob && pt.prob > (iter+1)->prob) {
                    priority.emplace(iter+1, pt); break;
                }
            }
            if (iter == priority.end()-1) { priority.emplace_back(pt); break; }
            if (iter == priority.begin() && iter->prob < pt.prob) {
                priority.emplace(iter, pt); break;
            }
        }
    }
    priority.erase(priority.begin());
}

vector<PT> PT::NewPTs() {
    vector<PT> res;
    if (content.size() == 1) return res;
    int init_pivot = pivot;
    for (int i = pivot; i < (int)curr_indices.size()-1; i++) {
        curr_indices[i]++;
        if (curr_indices[i] < max_indices[i]) { pivot = i; res.emplace_back(*this); }
        curr_indices[i]--;
    }
    pivot = init_pivot;
    return res;
}
