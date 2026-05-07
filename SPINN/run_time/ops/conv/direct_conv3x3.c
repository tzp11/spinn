/*
 * direct_conv3x3.c - 3×3 stride=1 直接卷积（无 im2col）
 *
 * 针对 3×3 stride=1 的常见情况，避免 im2col 的内存开销
 */

#include "direct_conv3x3.h"
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

/*
 * direct_conv3x3_s1 - 3×3 stride=1 直接卷积
 *
 * 输入: X[C][H][W]
 * 权重: W[OC][C][3][3]
 * 输出: Y[OC][OH][OW]
 */
void direct_conv3x3_s1(
    const float *X, int C, int H, int W,
    const float *weight, int OC,
    const float *bias,
    int PH, int PW,
    float *Y, int OH, int OW
) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(OC * OH * OW >= 4096)
    #endif
    for (int oc = 0; oc < OC; oc++) {
        const float *woc = weight + oc * C * 9;  /* 3×3 = 9 */
        float b = bias ? bias[oc] : 0.0f;
        float *yoc = Y + oc * OH * OW;
        
        /* 初始化为 bias */
        for (int i = 0; i < OH * OW; i++) {
            yoc[i] = b;
        }
        
        /* 累加所有输入通道 */
        for (int ic = 0; ic < C; ic++) {
            const float *xic = X + ic * H * W;
            const float *wic = woc + ic * 9;
            
            // Removed unused variables
            /* 遍历输出位置 */
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    float sum = 0.0f;
                    
                    /* 3×3 窗口 */
                    for (int kh = 0; kh < 3; kh++) {
                        int ih = oh + kh - PH;
                        if (ih < 0 || ih >= H) continue;
                        
                        for (int kw = 0; kw < 3; kw++) {
                            int iw = ow + kw - PW;
                            if (iw < 0 || iw >= W) continue;
                            
                            float x_val = xic[ih * W + iw];
                            float w_val = wic[kh * 3 + kw];
                            sum += x_val * w_val;
                        }
                    }
                    
                    yoc[oh * OW + ow] += sum;
                }
            }
        }
    }
}
