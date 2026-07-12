#ifndef MD5_H
#define MD5_H

#include <iostream>
#include <string>
#include <cstring>

using namespace std;

typedef unsigned char  Byte;
typedef unsigned int   bit32;

// MD5 轮转常量
#define S11 7
#define S12 12
#define S13 17
#define S14 22
#define S21 5
#define S22 9
#define S23 14
#define S24 20
#define S31 4
#define S32 11
#define S33 16
#define S34 23
#define S41 6
#define S42 10
#define S43 15
#define S44 21

// ============================================================
//  MD5 基本逻辑函数 — 标量版本
//  这些函数是 SIMD 并行化的主要目标
// ============================================================
#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z)))
#define ROTATELEFT(num, n) (((num) << (n)) | ((num) >> (32-(n))))

#define FF(a, b, c, d, x, s, ac) { \
    (a) += F((b), (c), (d)) + (x) + ac; \
    (a) = ROTATELEFT((a), (s)); \
    (a) += (b); \
}
#define GG(a, b, c, d, x, s, ac) { \
    (a) += G((b), (c), (d)) + (x) + ac; \
    (a) = ROTATELEFT((a), (s)); \
    (a) += (b); \
}
#define HH(a, b, c, d, x, s, ac) { \
    (a) += H((b), (c), (d)) + (x) + ac; \
    (a) = ROTATELEFT((a), (s)); \
    (a) += (b); \
}
#define II(a, b, c, d, x, s, ac) { \
    (a) += I((b), (c), (d)) + (x) + ac; \
    (a) = ROTATELEFT((a), (s)); \
    (a) += (b); \
}

// ---- 函数声明 ----
Byte* StringProcess(string input, int *n_byte);

// 串行版本 — 单条口令
void MD5Hash_serial(string input, bit32 *state);

// NEON SIMD 版本 — 同时计算 4 条口令
void MD5Hash_4x(string inputs[4], bit32 states[4][4]);

#endif
