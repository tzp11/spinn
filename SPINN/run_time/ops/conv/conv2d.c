#include "../kernels.h"
#include "../../../ennf_op_params.h"
#include "../mm/gemm_kernel.h"
#include "winograd.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ============================================================
 * im2col: 将输入按滑动窗口展开
 * 参照 ORT: onnxruntime/core/util/math/im2col.cc
 *
 * 输出: col[C*KH*KW, OH*OW]  (row = kernel element, col = output pixel)
 * ============================================================ */
static void im2col_cpu(const float *X, int C, int H, int W,
                       int KH, int KW, int SH, int SW, int PH, int PW,
                       int OH, int OW, float *col) {
    const int ohw = OH * OW;
    int row = 0;
    for (int c = 0; c < C; c++) {
        const float *xc = X + c * H * W;
        for (int kh = 0; kh < KH; kh++) {
            for (int kw = 0; kw < KW; kw++) {
                float *dst = col + row * ohw;
                for (int oh = 0; oh < OH; oh++) {
                    int ih = oh * SH - PH + kh;
                    if (ih < 0 || ih >= H) {
                        memset(dst + oh * OW, 0, OW * sizeof(float));
                    } else {
                        const float *xr = xc + ih * W;
                        for (int ow = 0; ow < OW; ow++) {
                            int iw = ow * SW - PW + kw;
                            dst[oh * OW + ow] = (iw >= 0 && iw < W) ? xr[iw] : 0.0f;
                        }
                    }
                }
                row++;
            }
        }
    }
}



/* ============================================================
 * Conv2D = im2col + GEMM (参照 ORT conv.cc)
 *
 * 关键路径：
 *   1. 1×1 卷积 (is_1x1): 完全跳过 im2col，直接 GEMM
 *      X[C,OH*OW] 本身即为 col 矩阵（已连续）
 *   2. 通用卷积: im2col + GEMM
 *      支持 OpenMP 并行 im2col (按 input channel 分段)
 * ============================================================ */
int op_conv2d(SpinnTensor **in, int n_in,
              void *params, uint16_t params_size,
              SpinnTensor **out, int n_out) {
    if (n_in < 2 || n_out < 1) return -1;

    float *X     = (float*)in[0]->data;
    float *W     = (float*)in[1]->data;
    float *Y     = (float*)out[0]->data;
    float *bias  = (n_in > 2 && in[2] && in[2]->data) ? (float*)in[2]->data : NULL;

    int N     = in[0]->dims[0];
    int C     = in[0]->dims[1];
    int H     = in[0]->dims[2];
    int Wdim  = in[0]->dims[3];
    int OC    = in[1]->dims[0];
    int KC    = in[1]->dims[1];

    int KH = in[1]->dims[2];
    int KW = in[1]->dims[3];
    int SH = 1, SW = 1, PH = 0, PW = 0;
    int group = 1, fused_act = 0;

    if (params && params_size >= sizeof(ENNF_ConvParams)) {
        ENNF_ConvParams *p = (ENNF_ConvParams*)params;
        if (p->group > 0)          group = p->group;
        if (p->kernel_shape[0] > 0) KH   = p->kernel_shape[0];
        if (p->kernel_shape[1] > 0) KW   = p->kernel_shape[1];
        if (p->strides[0] > 0)     SH   = p->strides[0];
        if (p->strides[1] > 0)     SW   = p->strides[1];
        PH = p->pads[0]; PW = p->pads[1];
        fused_act = p->reserved;

        /* SAME padding 计算 */
        int OH = out[0]->dims[2], OW = out[0]->dims[3];
        if (p->auto_pad == 1 || p->auto_pad == 2) {
            int ph = (OH-1)*SH + KH - H; if (ph < 0) ph = 0;
            int pw = (OW-1)*SW + KW - Wdim; if (pw < 0) pw = 0;
            PH = (p->auto_pad == 1) ? ph/2 : (ph+1)/2;
            PW = (p->auto_pad == 1) ? pw/2 : (pw+1)/2;
        }
    }

    int OH = out[0]->dims[2];
    int OW = out[0]->dims[3];
    int OC_g = OC / group;
    int kernel_dim = KC * KH * KW;
    int ohw = OH * OW;

    /* 1×1 无 padding 无 stride 的卷积判定 */
    int is_1x1 = (KH == 1 && KW == 1 && SH == 1 && SW == 1 && PH == 0 && PW == 0);

    /* Winograd F(2,3): 暂时禁用（浮点累积顺序不同导致输出偏差）
     * 算法实现正确，但浮点结果与 im2col 路径存在可见差异（未来可验证后启用）*/
    int use_winograd = 0;

    if (use_winograd) {
        int ret = 0;
        for (int n = 0; n < N; n++) {
            ret = winograd_conv_3x3(
                X + n * C * H * Wdim, 1, C, H, Wdim,
                W, OC, bias, PH, PW,
                Y + n * OC * ohw, OH, OW, fused_act);
            if (ret != 0) { use_winograd = 0; break; }
        }
        if (use_winograd) return 0;
        /* 失败则 fall through 到 im2col 路径 */
    }


    /* 分配 col 缓冲（1×1 时不需要） */
    float *col_buf = NULL;
    if (!is_1x1) {
        col_buf = (float*)malloc((size_t)kernel_dim * ohw * sizeof(float));
        if (!col_buf) return -1;
    }

    for (int n = 0; n < N; n++) {
        for (int g = 0; g < group; g++) {
            const float *Xg = X + (n * C + g * KC) * H * Wdim;
            const float *Wg = W + g * OC_g * kernel_dim;
            float *Yg = Y + (n * OC + g * OC_g) * ohw;

            const float *gemm_b;
            int ldb;

            if (is_1x1) {
                /* ---- 1×1 特化：直接用 X 做 GEMM 的 B ---- */
                /* X[KC, H*W] 已经是 col 矩阵 */
                gemm_b = Xg;
                ldb = ohw;
            } else {
                im2col_cpu(Xg, KC, H, Wdim, KH, KW, SH, SW, PH, PW, OH, OW, col_buf);

                gemm_b = col_buf;
                ldb = ohw;
            }

            /* Offline Pack Weights (Lazy Init) */
            if (!in[1]->packed_data) {
                size_t dummy_sz;
                /* Note: if group != 1, we must pack them group by group into a single buffer */
                /* For simplicity, we only trigger offline pack for group==1 right now. */
                if (group == 1) {
                    in[1]->packed_data = sgemm_pack_a_offline(OC_g, kernel_dim, Wg, kernel_dim, &dummy_sz);
                }
            }

            /* ---- GEMM ---- */
            if (in[1]->packed_data && group == 1) {
                sgemm_nn_packed_a(OC_g, ohw, kernel_dim,
                                  in[1]->packed_data,
                                  gemm_b, ldb,
                                  Yg, ohw);
            } else {
                sgemm_nn(OC_g, ohw, kernel_dim,
                         Wg, kernel_dim,
                         gemm_b, ldb,
                         Yg, ohw);
            }

            /* Bias */
            if (bias) {
                int oc_base = g * OC_g;
                for (int oc = 0; oc < OC_g; oc++) {
                    float b = bias[oc_base + oc];
                    float *yo = Yg + oc * ohw;
                    for (int i = 0; i < ohw; i++) yo[i] += b;
                }
            }
        }
    }

    if (col_buf) free(col_buf);

    /* 融合激活 */
    if (fused_act == 1) {
        int total = N * OC * OH * OW;
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(total >= 4096)
        #endif
        for (int i = 0; i < total; i++) {
            if (Y[i] < 0.0f) Y[i] = 0.0f;
        }
    }

    return 0;
}
