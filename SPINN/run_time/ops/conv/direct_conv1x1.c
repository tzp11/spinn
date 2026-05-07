/*
 * direct_conv1x1.c - 1×1 卷积的直接实现（无 im2col）
 * 
 * 参考 ORT: onnxruntime/core/mlas/lib/convolve.cpp
 * 
 * 优化策略:
 * 1. 完全跳过 im2col（零内存分配）
 * 2. 按 output channel 并行
 * 3. AVX2 SIMD 向量化
 * 4. 循环展开 + prefetch
 */

#include "../kernels.h"
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
#include <math.h>

/* ============================================================
 * Direct Conv 1×1 (单个 output channel)
 * ============================================================ */
static void direct_conv1x1_channel(
    const float *X,     // [C, H, W]
    const float *W,     // [C]
    float *Y,           // [H, W]
    int C, int HW,
    float bias
) {
    int hw = 0;
    
#if USE_AVX2
    const __m256 vbias = _mm256_set1_ps(bias);
    
    /* AVX2 路径: 每次处理 8 个像素 */
    for (; hw + 7 < HW; hw += 8) {
        __m256 sum = vbias;
        
        /* 内循环: 累加所有 input channels */
        for (int c = 0; c < C; c++) {
            __m256 x = _mm256_loadu_ps(&X[c * HW + hw]);
            __m256 w = _mm256_set1_ps(W[c]);
            sum = _mm256_fmadd_ps(x, w, sum);
        }
        
        /* 存储 */
        _mm256_storeu_ps(&Y[hw], sum);
    }
#endif
    
    /* 标量 fallback: 处理剩余像素 */
    for (; hw < HW; hw++) {
        float sum = bias;
        for (int c = 0; c < C; c++) {
            sum += X[c * HW + hw] * W[c];
        }
        Y[hw] = sum;
    }
}

/* ============================================================
 * Direct Conv 1×1 (单个 output channel) with Fused ReLU
 * ============================================================ */
static void direct_conv1x1_channel_relu(
    const float *X,     // [C, H, W]
    const float *W,     // [C]
    float *Y,           // [H, W]
    int C, int HW,
    float bias
) {
    int hw = 0;
    
#if USE_AVX2
    const __m256 zero = _mm256_setzero_ps();
    const __m256 vbias = _mm256_set1_ps(bias);
    
    /* AVX2 路径: 每次处理 8 个像素 */
    for (; hw + 7 < HW; hw += 8) {
        __m256 sum = vbias;
        
        /* 内循环: 累加所有 input channels */
        for (int c = 0; c < C; c++) {
            __m256 x = _mm256_loadu_ps(&X[c * HW + hw]);
            __m256 w = _mm256_set1_ps(W[c]);
            sum = _mm256_fmadd_ps(x, w, sum);
        }
        
        /* 融合 ReLU: max(0, sum) */
        sum = _mm256_max_ps(sum, zero);
        
        /* 存储 */
        _mm256_storeu_ps(&Y[hw], sum);
    }
#endif
    
    /* 标量 fallback: 处理剩余像素 */
    for (; hw < HW; hw++) {
        float sum = bias;
        for (int c = 0; c < C; c++) {
            sum += X[c * HW + hw] * W[c];
        }
        Y[hw] = (sum > 0.0f) ? sum : 0.0f;
    }
}

/* ============================================================
 * Direct Conv 1×1 (多个 output channels, 并行)
 * 
 * 输入: X[C, H, W]
 * 权重: W[OC, C]
 * 输出: Y[OC, H, W]
 * Bias: bias[OC] (可选)
 * ============================================================ */
void direct_conv1x1(
    const float *X,     // [C, H, W]
    const float *W,     // [OC, C]
    float *Y,           // [OC, H, W]
    const float *bias,  // [OC] or NULL
    int C, int H, int Wdim, int OC
) {
    const int HW = H * Wdim;
    
    /* 按 output channel 并行 */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(OC >= 4)
    #endif
    for (int oc = 0; oc < OC; oc++) {
        const float *Wc = W + oc * C;
        float *Yc = Y + oc * HW;
        float b = bias ? bias[oc] : 0.0f;
        
        direct_conv1x1_channel(X, Wc, Yc, C, HW, b);
    }
}

/* ============================================================
 * Direct Conv 1×1 with Fused Activation (优化版)
 * 
 * fused_act:
 *   0 = None
 *   1 = ReLU (在内核中融合)
 *   3 = SiLU
 * ============================================================ */
void direct_conv1x1_fused(
    const float *X,     // [C, H, W]
    const float *W,     // [OC, C]
    float *Y,           // [OC, H, W]
    const float *bias,  // [OC] or NULL
    int C, int H, int Wdim, int OC,
    int fused_act
) {
    const int HW = H * Wdim;
    
    if (fused_act == 1) {
        /* ReLU 融合路径 - 在内核中直接计算 */
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(OC >= 4)
        #endif
        for (int oc = 0; oc < OC; oc++) {
            const float *Wc = W + oc * C;
            float *Yc = Y + oc * HW;
            float b = bias ? bias[oc] : 0.0f;
            
            direct_conv1x1_channel_relu(X, Wc, Yc, C, HW, b);
        }
    } else {
        /* 标准路径 */
        direct_conv1x1(X, W, Y, bias, C, H, Wdim, OC);
        
        /* 其他激活函数 */
        if (fused_act == 3) {
            /* SiLU: y = x / (1 + exp(-x)) */
            const int total = OC * HW;
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static) if(total >= 4096)
            #endif
            for (int i = 0; i < total; i++) {
                float x = Y[i];
                Y[i] = x / (1.0f + expf(-x));
            }
        }
    }
}
