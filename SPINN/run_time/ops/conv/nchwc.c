/*
 * nchwc.c - NCHWc layout 转换和卷积（参考 MLAS）
 *
 * NCHWc layout: [N][C/8][H][W][8]
 * 优势: 8 个通道连续存储，完美适配 AVX2 (256bit = 8 float)
 */

#include "nchwc.h"
#include <string.h>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define USE_AVX2 1
#else
#define USE_AVX2 0
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

/* ============================================================
 * NCHW -> NCHWc 转换（参考 MLAS MlasReorderInputNchw）
 * ============================================================ */
void nchwc_reorder_input(
    const float *src,      /* NCHW: [C][H][W] */
    float *dst,            /* NCHWc: [C/8][H][W][8] */
    int C, int H, int W
) {
    const int HW = H * W;
    const int BlockSize = NCHWC_BLOCK_SIZE;
    
#if USE_AVX2
    const __m256 zero_vec = _mm256_setzero_ps();
#endif
    
    /* 按 8 个通道一组处理 */
    for (int c_block = 0; c_block < C; c_block += BlockSize) {
        int c_count = (C - c_block < BlockSize) ? (C - c_block) : BlockSize;
        
        /* 遍历空间位置 */
        for (int hw = 0; hw < HW; hw++) {
            const float *s = src + c_block * HW + hw;
            float *d = dst + (c_block / BlockSize) * HW * BlockSize + hw * BlockSize;
            
            /* 复制有效通道 */
            int c = 0;
            for (; c < c_count; c++) {
                d[c] = s[c * HW];
            }
            
            /* 填充零 */
#if USE_AVX2
            if (c < BlockSize) {
                _mm256_storeu_ps(d + c, zero_vec);
            }
#else
            for (; c < BlockSize; c++) {
                d[c] = 0.0f;
            }
#endif
        }
    }
}

/* ============================================================
 * NCHWc -> NCHW 转换（参考 MLAS MlasReorderOutputNchw）
 * ============================================================ */
void nchwc_reorder_output(
    const float *src,      /* NCHWc: [C/8][H][W][8] */
    float *dst,            /* NCHW: [C][H][W] */
    int C, int H, int W
) {
    const int HW = H * W;
    const int BlockSize = NCHWC_BLOCK_SIZE;
    
    /* 按 8 个通道一组处理 */
    for (int c_block = 0; c_block < C; c_block += BlockSize) {
        int c_count = (C - c_block < BlockSize) ? (C - c_block) : BlockSize;
        
        /* 遍历空间位置 */
        for (int hw = 0; hw < HW; hw++) {
            const float *s = src + (c_block / BlockSize) * HW * BlockSize + hw * BlockSize;
            float *d = dst + c_block * HW + hw;
            
            /* 复制有效通道 */
            for (int c = 0; c < c_count; c++) {
                d[c * HW] = s[c];
            }
        }
    }
}

/* ============================================================
 * NCHWc 1×1 卷积（参考 MLAS）
 * 
 * 输入: [IC/8][H][W][8]
 * 权重: [OC/8][IC/8][8][8]
 * 输出: [OC/8][H][W][8]
 * ============================================================ */
void nchwc_conv1x1(
    const float *input,
    const float *weight,
    const float *bias,
    float *output,
    int IC, int OC, int H, int W
) {
    const int HW = H * W;
    const int BlockSize = NCHWC_BLOCK_SIZE;
    const int IC_blocks = (IC + BlockSize - 1) / BlockSize;
    const int OC_blocks = (OC + BlockSize - 1) / BlockSize;
    
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(OC_blocks * HW >= 256)
    #endif
    for (int oc_blk = 0; oc_blk < OC_blocks; oc_blk++) {
        for (int hw = 0; hw < HW; hw++) {
            float *out = output + oc_blk * HW * BlockSize + hw * BlockSize;
            
#if USE_AVX2
            /* 初始化为 bias */
            __m256 acc = bias ? _mm256_loadu_ps(bias + oc_blk * BlockSize) : _mm256_setzero_ps();
            
            /* 累加所有输入通道块 */
            for (int ic_blk = 0; ic_blk < IC_blocks; ic_blk++) {
                const float *in = input + ic_blk * HW * BlockSize + hw * BlockSize;
                const float *w = weight + (oc_blk * IC_blocks + ic_blk) * BlockSize * BlockSize;
                
                /* 8×8 矩阵向量乘法 */
                for (int i = 0; i < BlockSize; i++) {
                    __m256 w_vec = _mm256_loadu_ps(w + i * BlockSize);
                    __m256 in_broadcast = _mm256_set1_ps(in[i]);
                    acc = _mm256_fmadd_ps(w_vec, in_broadcast, acc);
                }
            }
            
            _mm256_storeu_ps(out, acc);
#else
            /* 标量版本 */
            for (int i = 0; i < BlockSize; i++) {
                float sum = bias ? bias[oc_blk * BlockSize + i] : 0.0f;
                
                for (int ic_blk = 0; ic_blk < IC_blocks; ic_blk++) {
                    const float *in = input + ic_blk * HW * BlockSize + hw * BlockSize;
                    const float *w = weight + (oc_blk * IC_blocks + ic_blk) * BlockSize * BlockSize;
                    
                    for (int j = 0; j < BlockSize; j++) {
                        sum += w[i * BlockSize + j] * in[j];
                    }
                }
                
                out[i] = sum;
            }
#endif
        }
    }
}
