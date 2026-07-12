#include "../common/md5.h"
#include <arm_neon.h>
#include <cstdint>

// ============================================================
//  SIMD (ARM NEON) 加速版 MD5 — 一次并行计算 4 条口令
//
//  核心思路：
//    - 使用 uint32x4_t (128-bit NEON 寄存器) 同时处理 4 条口令
//    - F/G/H/I 四个基本逻辑函数全部向量化
//    - ROTATELEFT 用 vshlq + vsliq 实现循环左移
//    - FF/GG/HH/II 每轮操作也是 4 路并行
//
//  性能：理论加速比约 3-4× (受限于内存带宽和指令延迟)
// ============================================================

// ---- NEON 向量化的 MD5 基本函数 ----
#define F_NEON(x, y, z)  vorrq_u32(vandq_u32(x, y), vandq_u32(vmvnq_u32(x), z))
#define G_NEON(x, y, z)  vorrq_u32(vandq_u32(x, z), vandq_u32(y, vmvnq_u32(z)))
#define H_NEON(x, y, z)  veorq_u32(veorq_u32(x, y), z)
#define I_NEON(x, y, z)  veorq_u32(y, vorrq_u32(x, vmvnq_u32(z)))

#define ROTLEFT_NEON(a, n) \
    vorrq_u32(vshlq_u32(a, vdupq_n_s32(n)), vshlq_u32(a, vdupq_n_s32(n-32)))

#define FF_NEON(a,b,c,d,x,s,ac) { \
    a = vaddq_u32(a, F_NEON(b,c,d)); \
    a = vaddq_u32(a, x); \
    a = vaddq_u32(a, vdupq_n_u32(ac)); \
    a = ROTLEFT_NEON(a, s); \
    a = vaddq_u32(a, b); \
}
#define GG_NEON(a,b,c,d,x,s,ac) { \
    a = vaddq_u32(a, G_NEON(b,c,d)); \
    a = vaddq_u32(a, x); \
    a = vaddq_u32(a, vdupq_n_u32(ac)); \
    a = ROTLEFT_NEON(a, s); \
    a = vaddq_u32(a, b); \
}
#define HH_NEON(a,b,c,d,x,s,ac) { \
    a = vaddq_u32(a, H_NEON(b,c,d)); \
    a = vaddq_u32(a, x); \
    a = vaddq_u32(a, vdupq_n_u32(ac)); \
    a = ROTLEFT_NEON(a, s); \
    a = vaddq_u32(a, b); \
}
#define II_NEON(a,b,c,d,x,s,ac) { \
    a = vaddq_u32(a, I_NEON(b,c,d)); \
    a = vaddq_u32(a, x); \
    a = vaddq_u32(a, vdupq_n_u32(ac)); \
    a = ROTLEFT_NEON(a, s); \
    a = vaddq_u32(a, b); \
}

// ============================================================
//  MD5Hash_4x — NEON SIMD 批量 MD5 哈希
//  @param inputs[4]   4 条待哈希口令
//  @param states[4][4] 输出：4×4 个 32-bit 状态字
//
//  设计关键：
//    - 同一时刻，4 条口令各自占据 NEON 寄存器的一个 lane
//    - uint32x4_t 的 lane[0..3] 分别对应 inputs[0..3]
//    - 64 轮 MD5 变换在 4 条口令上完全同步执行
// ============================================================
void MD5Hash_4x(string inputs[4], bit32 states[4][4])
{
    // Step 1: 分别预处理 4 条口令（填充对齐）
    Byte* msg[4];
    int mlen[4];
    for (int i = 0; i < 4; i++)
        msg[i] = StringProcess(inputs[i], &mlen[i]);

    int blocks = mlen[0] / 64;   // 所有口令长度应相同（或取首条）

    // Step 2: 初始化 4 路 MD5 状态
    uint32x4_t h0 = vdupq_n_u32(0x67452301);
    uint32x4_t h1 = vdupq_n_u32(0xefcdab89);
    uint32x4_t h2 = vdupq_n_u32(0x98badcfe);
    uint32x4_t h3 = vdupq_n_u32(0x10325476);

    // Step 3: 逐 block 计算（每条口令的对应 block 同步处理）
    for (int blk = 0; blk < blocks; blk++)
    {
        // 装载 16 个 32-bit 字，每个字包含 4 条口令的对应数据
        uint32x4_t w[16];
        for (int k = 0; k < 16; k++)
        {
            uint32_t tmp[4];
            for (int idx = 0; idx < 4; idx++)
            {
                tmp[idx] = msg[idx][blk*64 + k*4]         |
                          (msg[idx][blk*64 + k*4 + 1] << 8)  |
                          (msg[idx][blk*64 + k*4 + 2] << 16) |
                          (msg[idx][blk*64 + k*4 + 3] << 24);
            }
            w[k] = vld1q_u32(tmp);
        }

        uint32x4_t a = h0, b = h1, c = h2, d = h3;

        // Round 1 — 4 路并行 FF
        FF_NEON(a,b,c,d,w[0], 7, 0xd76aa478); FF_NEON(d,a,b,c,w[1], 12,0xe8c7b756);
        FF_NEON(c,d,a,b,w[2], 17,0x242070db); FF_NEON(b,c,d,a,w[3], 22,0xc1bdceee);
        FF_NEON(a,b,c,d,w[4], 7, 0xf57c0faf); FF_NEON(d,a,b,c,w[5], 12,0x4787c62a);
        FF_NEON(c,d,a,b,w[6], 17,0xa8304613); FF_NEON(b,c,d,a,w[7], 22,0xfd469501);
        FF_NEON(a,b,c,d,w[8], 7, 0x698098d8); FF_NEON(d,a,b,c,w[9], 12,0x8b44f7af);
        FF_NEON(c,d,a,b,w[10],17,0xffff5bb1); FF_NEON(b,c,d,a,w[11],22,0x895cd7be);
        FF_NEON(a,b,c,d,w[12],7, 0x6b901122); FF_NEON(d,a,b,c,w[13],12,0xfd987193);
        FF_NEON(c,d,a,b,w[14],17,0xa679438e); FF_NEON(b,c,d,a,w[15],22,0x49b40821);

        // Round 2 — 4 路并行 GG
        GG_NEON(a,b,c,d,w[1], 5, 0xf61e2562); GG_NEON(d,a,b,c,w[6], 9, 0xc040b340);
        GG_NEON(c,d,a,b,w[11],14,0x265e5a51); GG_NEON(b,c,d,a,w[0], 20,0xe9b6c7aa);
        GG_NEON(a,b,c,d,w[5], 5, 0xd62f105d); GG_NEON(d,a,b,c,w[10],9, 0x02441453);
        GG_NEON(c,d,a,b,w[15],14,0xd8a1e681); GG_NEON(b,c,d,a,w[4], 20,0xe7d3fbc8);
        GG_NEON(a,b,c,d,w[9], 5, 0x21e1cde6); GG_NEON(d,a,b,c,w[14],9, 0xc33707d6);
        GG_NEON(c,d,a,b,w[3], 14,0xf4d50d87); GG_NEON(b,c,d,a,w[8], 20,0x455a14ed);
        GG_NEON(a,b,c,d,w[13],5, 0xa9e3e905); GG_NEON(d,a,b,c,w[2], 9, 0xfcefa3f8);
        GG_NEON(c,d,a,b,w[7], 14,0x676f02d9); GG_NEON(b,c,d,a,w[12],20,0x8d2a4c8a);

        // Round 3 — 4 路并行 HH
        HH_NEON(a,b,c,d,w[5], 4, 0xfffa3942); HH_NEON(d,a,b,c,w[8], 11,0x8771f681);
        HH_NEON(c,d,a,b,w[11],16,0x6d9d6122); HH_NEON(b,c,d,a,w[14],23,0xfde5380c);
        HH_NEON(a,b,c,d,w[1], 4, 0xa4beea44); HH_NEON(d,a,b,c,w[4], 11,0x4bdecfa9);
        HH_NEON(c,d,a,b,w[7], 16,0xf6bb4b60); HH_NEON(b,c,d,a,w[10],23,0xbebfbc70);
        HH_NEON(a,b,c,d,w[13],4, 0x289b7ec6); HH_NEON(d,a,b,c,w[0], 11,0xeaa127fa);
        HH_NEON(c,d,a,b,w[3], 16,0xd4ef3085); HH_NEON(b,c,d,a,w[6], 23,0x04881d05);
        HH_NEON(a,b,c,d,w[9], 4, 0xd9d4d039); HH_NEON(d,a,b,c,w[12],11,0xe6db99e5);
        HH_NEON(c,d,a,b,w[15],16,0x1fa27cf8); HH_NEON(b,c,d,a,w[2], 23,0xc4ac5665);

        // Round 4 — 4 路并行 II
        II_NEON(a,b,c,d,w[0], 6, 0xf4292244); II_NEON(d,a,b,c,w[7], 10,0x432aff97);
        II_NEON(c,d,a,b,w[14],15,0xab9423a7); II_NEON(b,c,d,a,w[5], 21,0xfc93a039);
        II_NEON(a,b,c,d,w[12],6, 0x655b59c3); II_NEON(d,a,b,c,w[3], 10,0x8f0ccc92);
        II_NEON(c,d,a,b,w[10],15,0xffeff47d); II_NEON(b,c,d,a,w[1], 21,0x85845dd1);
        II_NEON(a,b,c,d,w[8], 6, 0x6fa87e4f); II_NEON(d,a,b,c,w[15],10,0xfe2ce6e0);
        II_NEON(c,d,a,b,w[6], 15,0xa3014314); II_NEON(b,c,d,a,w[13],21,0x4e0811a1);
        II_NEON(a,b,c,d,w[4], 6, 0xf7537e82); II_NEON(d,a,b,c,w[11],10,0xbd3af235);
        II_NEON(c,d,a,b,w[2], 15,0x2ad7d2bb); II_NEON(b,c,d,a,w[9], 21,0xeb86d391);

        h0 = vaddq_u32(h0, a);
        h1 = vaddq_u32(h1, b);
        h2 = vaddq_u32(h2, c);
        h3 = vaddq_u32(h3, d);
    }

    // Step 4: 从 NEON 寄存器提取结果到 states[4][4]
    uint32_t r0[4], r1[4], r2[4], r3[4];
    vst1q_u32(r0, h0); vst1q_u32(r1, h1);
    vst1q_u32(r2, h2); vst1q_u32(r3, h3);

    for (int i = 0; i < 4; i++)
    {
        states[i][0] = ((r0[i] & 0xff) << 24)       |
                       ((r0[i] & 0xff00) << 8)       |
                       ((r0[i] & 0xff0000) >> 8)     |
                       ((r0[i] & 0xff000000) >> 24);
        states[i][1] = ((r1[i] & 0xff) << 24)       |
                       ((r1[i] & 0xff00) << 8)       |
                       ((r1[i] & 0xff0000) >> 8)     |
                       ((r1[i] & 0xff000000) >> 24);
        states[i][2] = ((r2[i] & 0xff) << 24)       |
                       ((r2[i] & 0xff00) << 8)       |
                       ((r2[i] & 0xff0000) >> 8)     |
                       ((r2[i] & 0xff000000) >> 24);
        states[i][3] = ((r3[i] & 0xff) << 24)       |
                       ((r3[i] & 0xff00) << 8)       |
                       ((r3[i] & 0xff0000) >> 8)     |
                       ((r3[i] & 0xff000000) >> 24);
    }

    for (int i = 0; i < 4; i++) delete[] msg[i];
}
