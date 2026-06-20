#include "md5.h"
#include <iomanip>
#include <assert.h>
#include <chrono>
#include <arm_neon.h>
#include <cstdint>
#include <omp.h>
#include <cstring>
using namespace std;
using namespace chrono;

/**
 * StringProcess: 将单个输入字符串转换成MD5计算所需的消息数组
 * @param input 输入
 * @param[out] n_byte 用于给调用者传递额外的返回值，即最终Byte数组的长度
 * @return Byte消息数组
 */
#define F_NEON(x, y, z) vorrq_u32(vandq_u32(x, y), vandq_u32(vmvnq_u32(x), z))
#define G_NEON(x, y, z) vorrq_u32(vandq_u32(x, z), vandq_u32(y, vmvnq_u32(z)))
#define H_NEON(x, y, z) veorq_u32(veorq_u32(x, y), z)
#define I_NEON(x, y, z) veorq_u32(y, vorrq_u32(x, vmvnq_u32(z)))
#define ROTLEFT_NEON(a, n) vorrq_u32(vshlq_u32(a, vdupq_n_s32(n)), vshlq_u32(a, vdupq_n_s32(n-32)))

#define FF_NEON(a, b, c, d, x, s, ac) { a = vaddq_u32(a, F_NEON(b, c, d)); a = vaddq_u32(a, x); a = vaddq_u32(a, vdupq_n_u32(ac)); a = ROTLEFT_NEON(a, s); a = vaddq_u32(a, b); }
#define GG_NEON(a, b, c, d, x, s, ac) { a = vaddq_u32(a, G_NEON(b, c, d)); a = vaddq_u32(a, x); a = vaddq_u32(a, vdupq_n_u32(ac)); a = ROTLEFT_NEON(a, s); a = vaddq_u32(a, b); }
#define HH_NEON(a, b, c, d, x, s, ac) { a = vaddq_u32(a, H_NEON(b, c, d)); a = vaddq_u32(a, x); a = vaddq_u32(a, vdupq_n_u32(ac)); a = ROTLEFT_NEON(a, s); a = vaddq_u32(a, b); }
#define II_NEON(a, b, c, d, x, s, ac) { a = vaddq_u32(a, I_NEON(b, c, d)); a = vaddq_u32(a, x); a = vaddq_u32(a, vdupq_n_u32(ac)); a = ROTLEFT_NEON(a, s); a = vaddq_u32(a, b); }
Byte *StringProcess(string input, int *n_byte)
{
	// 将输入的字符串转换为Byte为单位的数组
	Byte *blocks = (Byte *)input.c_str();
	int length = input.length();

	// 计算原始消息长度（以比特为单位）
	int bitLength = length * 8;

	// paddingBits: 原始消息需要的padding长度（以bit为单位）
	// 对于给定的消息，将其补齐至length%512==448为止
	// 需要注意的是，即便给定的消息满足length%512==448，也需要再pad 512bits
	int paddingBits = bitLength % 512;
	if (paddingBits > 448)
	{
		paddingBits = 512 - (paddingBits - 448);
	}
	else if (paddingBits < 448)
	{
		paddingBits = 448 - paddingBits;
	}
	else if (paddingBits == 448)
	{
		paddingBits = 512;
	}

	// 原始消息需要的padding长度（以Byte为单位）
	int paddingBytes = paddingBits / 8;
	// 创建最终的字节数组
	// length + paddingBytes + 8:
	// 1. length为原始消息的长度（bits）
	// 2. paddingBytes为原始消息需要的padding长度（Bytes）
	// 3. 在pad到length%512==448之后，需要额外附加64bits的原始消息长度，即8个bytes
	int paddedLength = length + paddingBytes + 8;
	Byte *paddedMessage = new Byte[paddedLength];

	// 复制原始消息
	memcpy(paddedMessage, blocks, length);

	// 添加填充字节。填充时，第一位为1，后面的所有位均为0。
	// 所以第一个byte是0x80
	paddedMessage[length] = 0x80;							 // 添加一个0x80字节
	memset(paddedMessage + length + 1, 0, paddingBytes - 1); // 填充0字节

	// 添加消息长度（64比特，小端格式）
	for (int i = 0; i < 8; ++i)
	{
		// 特别注意此处应当将bitLength转换为uint64_t
		// 这里的length是原始消息的长度
		paddedMessage[length + paddingBytes + i] = ((uint64_t)length * 8 >> (i * 8)) & 0xFF;
	}

	// 验证长度是否满足要求。此时长度应当是512bit的倍数
	int residual = 8 * paddedLength % 512;
	// assert(residual == 0);

	// 在填充+添加长度之后，消息被分为n_blocks个512bit的部分
	*n_byte = paddedLength;
	return paddedMessage;
}


/**
 * MD5Hash: 将单个输入字符串转换成MD5
 * @param input 输入
 * @param[out] state 用于给调用者传递额外的返回值，即最终的缓冲区，也就是MD5的结果
 * @return Byte消息数组
 */
// NEON 并行版本：同时计算 4 个口令
void MD5Hash(string inputs[4], bit32 states[4][4]) {
    Byte* msg[4];
    int mlen[4];
    for (int i = 0; i < 4; i++) {
        msg[i] = StringProcess(inputs[i], &mlen[i]);
    }
    int blocks = mlen[0] / 64;
    
    uint32x4_t h0 = vdupq_n_u32(0x67452301);
    uint32x4_t h1 = vdupq_n_u32(0xefcdab89);
    uint32x4_t h2 = vdupq_n_u32(0x98badcfe);
    uint32x4_t h3 = vdupq_n_u32(0x10325476);
    
    for (int blk = 0; blk < blocks; blk++) {
        uint32x4_t w[16];
        for (int k = 0; k < 16; k++) {
            uint32_t tmp[4];
            for (int idx = 0; idx < 4; idx++) {
                tmp[idx] = msg[idx][blk*64 + k*4] |
                          (msg[idx][blk*64 + k*4 + 1] << 8) |
                          (msg[idx][blk*64 + k*4 + 2] << 16) |
                          (msg[idx][blk*64 + k*4 + 3] << 24);
            }
            w[k] = vld1q_u32(tmp);
        }
        
        uint32x4_t a = h0, b = h1, c = h2, d = h3;
        
        FF_NEON(a, b, c, d, w[0], 7, 0xd76aa478);
        FF_NEON(d, a, b, c, w[1], 12, 0xe8c7b756);
        FF_NEON(c, d, a, b, w[2], 17, 0x242070db);
        FF_NEON(b, c, d, a, w[3], 22, 0xc1bdceee);
        FF_NEON(a, b, c, d, w[4], 7, 0xf57c0faf);
        FF_NEON(d, a, b, c, w[5], 12, 0x4787c62a);
        FF_NEON(c, d, a, b, w[6], 17, 0xa8304613);
        FF_NEON(b, c, d, a, w[7], 22, 0xfd469501);
        FF_NEON(a, b, c, d, w[8], 7, 0x698098d8);
        FF_NEON(d, a, b, c, w[9], 12, 0x8b44f7af);
        FF_NEON(c, d, a, b, w[10], 17, 0xffff5bb1);
        FF_NEON(b, c, d, a, w[11], 22, 0x895cd7be);
        FF_NEON(a, b, c, d, w[12], 7, 0x6b901122);
        FF_NEON(d, a, b, c, w[13], 12, 0xfd987193);
        FF_NEON(c, d, a, b, w[14], 17, 0xa679438e);
        FF_NEON(b, c, d, a, w[15], 22, 0x49b40821);
        
        GG_NEON(a, b, c, d, w[1], 5, 0xf61e2562);
        GG_NEON(d, a, b, c, w[6], 9, 0xc040b340);
        GG_NEON(c, d, a, b, w[11], 14, 0x265e5a51);
        GG_NEON(b, c, d, a, w[0], 20, 0xe9b6c7aa);
        GG_NEON(a, b, c, d, w[5], 5, 0xd62f105d);
        GG_NEON(d, a, b, c, w[10], 9, 0x02441453);
        GG_NEON(c, d, a, b, w[15], 14, 0xd8a1e681);
        GG_NEON(b, c, d, a, w[4], 20, 0xe7d3fbc8);
        GG_NEON(a, b, c, d, w[9], 5, 0x21e1cde6);
        GG_NEON(d, a, b, c, w[14], 9, 0xc33707d6);
        GG_NEON(c, d, a, b, w[3], 14, 0xf4d50d87);
        GG_NEON(b, c, d, a, w[8], 20, 0x455a14ed);
        GG_NEON(a, b, c, d, w[13], 5, 0xa9e3e905);
        GG_NEON(d, a, b, c, w[2], 9, 0xfcefa3f8);
        GG_NEON(c, d, a, b, w[7], 14, 0x676f02d9);
        GG_NEON(b, c, d, a, w[12], 20, 0x8d2a4c8a);
        
        HH_NEON(a, b, c, d, w[5], 4, 0xfffa3942);
        HH_NEON(d, a, b, c, w[8], 11, 0x8771f681);
        HH_NEON(c, d, a, b, w[11], 16, 0x6d9d6122);
        HH_NEON(b, c, d, a, w[14], 23, 0xfde5380c);
        HH_NEON(a, b, c, d, w[1], 4, 0xa4beea44);
        HH_NEON(d, a, b, c, w[4], 11, 0x4bdecfa9);
        HH_NEON(c, d, a, b, w[7], 16, 0xf6bb4b60);
        HH_NEON(b, c, d, a, w[10], 23, 0xbebfbc70);
        HH_NEON(a, b, c, d, w[13], 4, 0x289b7ec6);
        HH_NEON(d, a, b, c, w[0], 11, 0xeaa127fa);
        HH_NEON(c, d, a, b, w[3], 16, 0xd4ef3085);
        HH_NEON(b, c, d, a, w[6], 23, 0x04881d05);
        HH_NEON(a, b, c, d, w[9], 4, 0xd9d4d039);
        HH_NEON(d, a, b, c, w[12], 11, 0xe6db99e5);
        HH_NEON(c, d, a, b, w[15], 16, 0x1fa27cf8);
        HH_NEON(b, c, d, a, w[2], 23, 0xc4ac5665);
        
        II_NEON(a, b, c, d, w[0], 6, 0xf4292244);
        II_NEON(d, a, b, c, w[7], 10, 0x432aff97);
        II_NEON(c, d, a, b, w[14], 15, 0xab9423a7);
        II_NEON(b, c, d, a, w[5], 21, 0xfc93a039);
        II_NEON(a, b, c, d, w[12], 6, 0x655b59c3);
        II_NEON(d, a, b, c, w[3], 10, 0x8f0ccc92);
        II_NEON(c, d, a, b, w[10], 15, 0xffeff47d);
        II_NEON(b, c, d, a, w[1], 21, 0x85845dd1);
        II_NEON(a, b, c, d, w[8], 6, 0x6fa87e4f);
        II_NEON(d, a, b, c, w[15], 10, 0xfe2ce6e0);
        II_NEON(c, d, a, b, w[6], 15, 0xa3014314);
        II_NEON(b, c, d, a, w[13], 21, 0x4e0811a1);
        II_NEON(a, b, c, d, w[4], 6, 0xf7537e82);
        II_NEON(d, a, b, c, w[11], 10, 0xbd3af235);
        II_NEON(c, d, a, b, w[2], 15, 0x2ad7d2bb);
        II_NEON(b, c, d, a, w[9], 21, 0xeb86d391);
        
        h0 = vaddq_u32(h0, a);
        h1 = vaddq_u32(h1, b);
        h2 = vaddq_u32(h2, c);
        h3 = vaddq_u32(h3, d);
    }
    
    uint32_t r0[4], r1[4], r2[4], r3[4];    
    vst1q_u32(r0, h0);
    vst1q_u32(r1, h1);
    vst1q_u32(r2, h2);
    vst1q_u32(r3, h3);
    
    for (int i = 0; i < 4; i++) {
        states[i][0] = ((r0[i] & 0xff) << 24) | ((r0[i] & 0xff00) << 8) |
                       ((r0[i] & 0xff0000) >> 8) | ((r0[i] & 0xff000000) >> 24);
        states[i][1] = ((r1[i] & 0xff) << 24) | ((r1[i] & 0xff00) << 8) |
                       ((r1[i] & 0xff0000) >> 8) | ((r1[i] & 0xff000000) >> 24);
        states[i][2] = ((r2[i] & 0xff) << 24) | ((r2[i] & 0xff00) << 8) |
                       ((r2[i] & 0xff0000) >> 8) | ((r2[i] & 0xff000000) >> 24);
        states[i][3] = ((r3[i] & 0xff) << 24) | ((r3[i] & 0xff00) << 8) |
                       ((r3[i] & 0xff0000) >> 8) | ((r3[i] & 0xff000000) >> 24);
    }
    
    for (int i = 0; i < 4; i++) delete[] msg[i];
}