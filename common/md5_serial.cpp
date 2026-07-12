#include "md5.h"
#include <cstdint>
#include <cassert>

// ============================================================
//  StringProcess — 将输入字符串填充为 512-bit 对齐的消息块
//  MD5 消息填充规则：
//    1. 附加一个 1 bit (0x80)
//    2. 填充 0 直到 length % 512 == 448
//    3. 附加 64-bit 原始消息长度（小端序）
// ============================================================
Byte* StringProcess(string input, int *n_byte)
{
    Byte *blocks = (Byte*)input.c_str();
    int length = input.length();
    int bitLength = length * 8;

    // 计算需要的填充 bit 数
    int paddingBits = bitLength % 512;
    if (paddingBits > 448)
        paddingBits = 512 - (paddingBits - 448);
    else if (paddingBits < 448)
        paddingBits = 448 - paddingBits;
    else  // paddingBits == 448
        paddingBits = 512;

    int paddingBytes = paddingBits / 8;
    int paddedLength = length + paddingBytes + 8;  // +8 为 64-bit 长度字段
    Byte *paddedMessage = new Byte[paddedLength];

    memcpy(paddedMessage, blocks, length);
    paddedMessage[length] = 0x80;                         // 附加 0x80
    memset(paddedMessage + length + 1, 0, paddingBytes - 1);  // 填充 0

    // 附加原始消息长度 (小端序)
    for (int i = 0; i < 8; ++i)
        paddedMessage[length + paddingBytes + i] =
            ((uint64_t)length * 8 >> (i * 8)) & 0xFF;

    *n_byte = paddedLength;
    return paddedMessage;
}

// ============================================================
//  MD5Hash_serial — 串行 MD5，对单条口令计算哈希
//  作为所有并行版本的"黄金参考"（golden reference）
// ============================================================
void MD5Hash_serial(string input, bit32 *state)
{
    int messageLength;
    Byte *paddedMessage = StringProcess(input, &messageLength);
    int n_blocks = messageLength / 64;

    // MD5 初始 IV
    state[0] = 0x67452301;
    state[1] = 0xefcdab89;
    state[2] = 0x98badcfe;
    state[3] = 0x10325476;

    // 逐 512-bit block 更新
    for (int i = 0; i < n_blocks; i++)
    {
        bit32 x[16];
        // 从 Byte 数组装载为 32-bit 小端字
        for (int j = 0; j < 16; j++)
            x[j] = (paddedMessage[4*j   + i*64])       |
                   (paddedMessage[4*j+1 + i*64] << 8)  |
                   (paddedMessage[4*j+2 + i*64] << 16) |
                   (paddedMessage[4*j+3 + i*64] << 24);

        bit32 a = state[0], b = state[1], c = state[2], d = state[3];

        // Round 1
        FF(a,b,c,d, x[0], S11,0xd76aa478); FF(d,a,b,c, x[1], S12,0xe8c7b756);
        FF(c,d,a,b, x[2], S13,0x242070db); FF(b,c,d,a, x[3], S14,0xc1bdceee);
        FF(a,b,c,d, x[4], S11,0xf57c0faf); FF(d,a,b,c, x[5], S12,0x4787c62a);
        FF(c,d,a,b, x[6], S13,0xa8304613); FF(b,c,d,a, x[7], S14,0xfd469501);
        FF(a,b,c,d, x[8], S11,0x698098d8); FF(d,a,b,c, x[9], S12,0x8b44f7af);
        FF(c,d,a,b,x[10], S13,0xffff5bb1); FF(b,c,d,a,x[11], S14,0x895cd7be);
        FF(a,b,c,d,x[12], S11,0x6b901122); FF(d,a,b,c,x[13], S12,0xfd987193);
        FF(c,d,a,b,x[14], S13,0xa679438e); FF(b,c,d,a,x[15], S14,0x49b40821);

        // Round 2
        GG(a,b,c,d, x[1], S21,0xf61e2562); GG(d,a,b,c, x[6], S22,0xc040b340);
        GG(c,d,a,b,x[11], S23,0x265e5a51); GG(b,c,d,a, x[0], S24,0xe9b6c7aa);
        GG(a,b,c,d, x[5], S21,0xd62f105d); GG(d,a,b,c,x[10], S22,0x02441453);
        GG(c,d,a,b,x[15], S23,0xd8a1e681); GG(b,c,d,a, x[4], S24,0xe7d3fbc8);
        GG(a,b,c,d, x[9], S21,0x21e1cde6); GG(d,a,b,c,x[14], S22,0xc33707d6);
        GG(c,d,a,b, x[3], S23,0xf4d50d87); GG(b,c,d,a, x[8], S24,0x455a14ed);
        GG(a,b,c,d,x[13], S21,0xa9e3e905); GG(d,a,b,c, x[2], S22,0xfcefa3f8);
        GG(c,d,a,b, x[7], S23,0x676f02d9); GG(b,c,d,a,x[12], S24,0x8d2a4c8a);

        // Round 3
        HH(a,b,c,d, x[5], S31,0xfffa3942); HH(d,a,b,c, x[8], S32,0x8771f681);
        HH(c,d,a,b,x[11], S33,0x6d9d6122); HH(b,c,d,a,x[14], S34,0xfde5380c);
        HH(a,b,c,d, x[1], S31,0xa4beea44); HH(d,a,b,c, x[4], S32,0x4bdecfa9);
        HH(c,d,a,b, x[7], S33,0xf6bb4b60); HH(b,c,d,a,x[10], S34,0xbebfbc70);
        HH(a,b,c,d,x[13], S31,0x289b7ec6); HH(d,a,b,c, x[0], S32,0xeaa127fa);
        HH(c,d,a,b, x[3], S33,0xd4ef3085); HH(b,c,d,a, x[6], S34,0x04881d05);
        HH(a,b,c,d, x[9], S31,0xd9d4d039); HH(d,a,b,c,x[12], S32,0xe6db99e5);
        HH(c,d,a,b,x[15], S33,0x1fa27cf8); HH(b,c,d,a, x[2], S34,0xc4ac5665);

        // Round 4
        II(a,b,c,d, x[0], S41,0xf4292244); II(d,a,b,c, x[7], S42,0x432aff97);
        II(c,d,a,b,x[14], S43,0xab9423a7); II(b,c,d,a, x[5], S44,0xfc93a039);
        II(a,b,c,d,x[12], S41,0x655b59c3); II(d,a,b,c, x[3], S42,0x8f0ccc92);
        II(c,d,a,b,x[10], S43,0xffeff47d); II(b,c,d,a, x[1], S44,0x85845dd1);
        II(a,b,c,d, x[8], S41,0x6fa87e4f); II(d,a,b,c,x[15], S42,0xfe2ce6e0);
        II(c,d,a,b, x[6], S43,0xa3014314); II(b,c,d,a,x[13], S44,0x4e0811a1);
        II(a,b,c,d, x[4], S41,0xf7537e82); II(d,a,b,c,x[11], S42,0xbd3af235);
        II(c,d,a,b, x[2], S43,0x2ad7d2bb); II(b,c,d,a, x[9], S44,0xeb86d391);

        state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    }

    // 字节序转换（小端 → 大端显示）
    for (int i = 0; i < 4; i++)
    {
        uint32_t value = state[i];
        state[i] = ((value & 0xff) << 24)       |
                   ((value & 0xff00) << 8)       |
                   ((value & 0xff0000) >> 8)     |
                   ((value & 0xff000000) >> 24);
    }

    delete[] paddedMessage;
}
