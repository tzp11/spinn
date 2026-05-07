/* Test Winograd F(2,3) vs im2col+GEMM for a small 3x3 conv */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "ops/conv/winograd.h"
#include "ops/mm/gemm_kernel.h"

/* im2col + GEMM reference */
static void im2col_ref(const float *X, int C, int H, int W,
                       int KH, int KW, int PH, int PW,
                       int OH, int OW, float *col) {
    int ohw = OH * OW;
    int row = 0;
    for (int c = 0; c < C; c++) {
        const float *xc = X + c * H * W;
        for (int kh = 0; kh < KH; kh++) {
            for (int kw = 0; kw < KW; kw++) {
                float *dst = col + row * ohw;
                for (int oh = 0; oh < OH; oh++) {
                    int ih = oh - PH + kh;
                    for (int ow = 0; ow < OW; ow++) {
                        int iw = ow - PW + kw;
                        dst[oh * OW + ow] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                            ? xc[ih * W + iw] : 0.0f;
                    }
                }
                row++;
            }
        }
    }
}

/* Direct scalar 3x3 conv for verification */
static void conv3x3_ref(const float *X, int C, int H, int W,
                        const float *weight, int OC,
                        const float *bias, int PH, int PW,
                        float *Y, int OH, int OW) {
    for (int oc = 0; oc < OC; oc++) {
        for (int oh = 0; oh < OH; oh++) {
            for (int ow = 0; ow < OW; ow++) {
                float sum = bias ? bias[oc] : 0.0f;
                for (int ic = 0; ic < C; ic++) {
                    for (int kh = 0; kh < 3; kh++) {
                        for (int kw = 0; kw < 3; kw++) {
                            int ih = oh - PH + kh;
                            int iw = ow - PW + kw;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                sum += X[ic * H * W + ih * W + iw]
                                     * weight[oc * C * 9 + ic * 9 + kh * 3 + kw];
                            }
                        }
                    }
                }
                Y[oc * OH * OW + oh * OW + ow] = sum;
            }
        }
    }
}

int main() {
    /* Small test: C=2, H=4, W=4, OC=3, KH=KW=3, SH=SW=1, PH=PW=1 */
    int C = 2, H = 4, W = 4, OC = 3;
    int OH = H, OW = W;
    int ohw = OH * OW;
    int kernel_dim = C * 9;

    float X[2 * 4 * 4];
    for (int i = 0; i < 32; i++) X[i] = (float)(i + 1);

    float weight[3 * 2 * 9];
    for (int i = 0; i < 54; i++) weight[i] = (float)(i % 5 - 2) * 0.5f;

    float bias_ref[3] = {1.0f, 2.0f, 3.0f};

    /* === Direct conv (ground truth) === */
    float Y_direct[3 * 16];
    conv3x3_ref(X, C, H, W, weight, OC, bias_ref, 1, 1, Y_direct, OH, OW);

    /* === im2col + GEMM === */
    float col[18 * 16];
    float Y_gemm[3 * 16];
    im2col_ref(X, C, H, W, 3, 3, 1, 1, OH, OW, col);
    sgemm_nn(OC, ohw, kernel_dim, weight, kernel_dim, col, ohw, Y_gemm, ohw);
    for (int oc = 0; oc < OC; oc++)
        for (int i = 0; i < ohw; i++)
            Y_gemm[oc * ohw + i] += bias_ref[oc];

    /* === Manual Winograd (single tile, OC0, IC0) for debugging === */
    printf("\n--- Manual Winograd debug (tile 0, OC0) ---\n");
    /* Extract 4x4 input tile for IC=0, tile (0,0) */
    float d0[4][4];
    for (int i = 0; i < 4; i++) {
        int ih = i - 1; /* PH=1 */
        for (int j = 0; j < 4; j++) {
            int iw = j - 1;
            d0[i][j] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                       ? X[0 * H * W + ih * W + iw] : 0.0f;
        }
    }
    printf("Input tile IC0:\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) printf("%.1f ", d0[i][j]);
        printf("\n");
    }

    /* Weight for OC=0, IC=0: 3x3 */
    float g0[3][3];
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            g0[i][j] = weight[0 * C * 9 + 0 * 9 + i * 3 + j];
    printf("Weight OC0 IC0:\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) printf("%.2f ", g0[i][j]);
        printf("\n");
    }

    /* Direct compute for tile 0 output */
    float direct_tile[2][2] = {{0}};
    for (int ti = 0; ti < 2; ti++)
        for (int tj = 0; tj < 2; tj++) {
            float s = 0;
            for (int ic = 0; ic < C; ic++)
                for (int kh = 0; kh < 3; kh++)
                    for (int kw = 0; kw < 3; kw++) {
                        int ih = ti - 1 + kh, iw = tj - 1 + kw;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                            s += X[ic * H * W + ih * W + iw]
                               * weight[0 * C * 9 + ic * 9 + kh * 3 + kw];
                    }
            direct_tile[ti][tj] = s + bias_ref[0];
        }
    printf("Direct tile 0 OC0: %.2f %.2f / %.2f %.2f\n",
           direct_tile[0][0], direct_tile[0][1],
           direct_tile[1][0], direct_tile[1][1]);

    /* === Winograd === */
    float *U = winograd_pack_weight(weight, OC, C);
    float Y_win[3 * 16];
    int ret = winograd_conv_3x3(X, 1, C, H, W, U, OC, bias_ref, 1, 1, Y_win, OH, OW, 0);
    free(U);

    /* Compare all three */
    printf("ret = %d\n", ret);
    float max_diff_gemm = 0, max_diff_win = 0;
    for (int i = 0; i < OC * ohw; i++) {
        float dg = fabsf(Y_direct[i] - Y_gemm[i]);
        float dw = fabsf(Y_direct[i] - Y_win[i]);
        if (dg > max_diff_gemm) max_diff_gemm = dg;
        if (dw > max_diff_win) max_diff_win = dw;
    }
    printf("Max diff (direct vs gemm): %e\n", max_diff_gemm);
    printf("Max diff (direct vs wino):  %e\n", max_diff_win);

    printf("\nDirect: ");
    for (int i = 0; i < 8; i++) printf("%.3f ", Y_direct[i]);
    printf("\nGEMM:   ");
    for (int i = 0; i < 8; i++) printf("%.3f ", Y_gemm[i]);
    printf("\nWino:   ");
    for (int i = 0; i < 8; i++) printf("%.3f ", Y_win[i]);
    printf("\n\nAll direct values:\n");
    for (int oc = 0; oc < OC; oc++) {
        printf("OC%d: ", oc);
        for (int i = 0; i < ohw; i++) printf("%.2f ", Y_direct[oc*ohw+i]);
        printf("\n");
    }
    printf("\nAll wino values:\n");
    for (int oc = 0; oc < OC; oc++) {
        printf("OC%d: ", oc);
        for (int i = 0; i < ohw; i++) printf("%.2f ", Y_win[oc*ohw+i]);
        printf("\n");
    }

    return max_diff_win > 0.1f ? 1 : 0;
}
