/*
 * winograd.c - Winograd F(2×2, 3×3) - GEMM-based 高效实现
 *
 * 优化: 权重变换缓存 + 线程局部 V/M 缓冲复用
 */

#include "winograd.h"
#include "../mm/gemm_kernel.h"
#include <string.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define ALPHA 4
#define ATILES 16
#define TILE_M 2

/* ---- 变换矩阵 ---- */

/* B^T × d × B: 输入变换 4×4 → 4×4 */
static inline void transform_input(const float d[4][4], float V[4][4]) {
    float t[4][4];
    for (int j = 0; j < 4; j++) {
        t[0][j] =  d[0][j] - d[2][j];
        t[1][j] =  d[1][j] + d[2][j];
        t[2][j] = -d[1][j] + d[2][j];
        t[3][j] =  d[1][j] - d[3][j];
    }
    for (int i = 0; i < 4; i++) {
        V[i][0] =  t[i][0] - t[i][2];
        V[i][1] =  t[i][1] + t[i][2];
        V[i][2] = -t[i][1] + t[i][2];
        V[i][3] =  t[i][1] - t[i][3];
    }
}

/* G × g × G^T: 权重变换 3×3 → 4×4 */
static inline void transform_weight(const float g[3][3], float U[4][4]) {
    float t[4][3];
    for (int j = 0; j < 3; j++) {
        t[0][j] =         g[0][j];
        t[1][j] = 0.5f * (g[0][j] + g[1][j] + g[2][j]);
        t[2][j] = 0.5f * (g[0][j] - g[1][j] + g[2][j]);
        t[3][j] =         g[2][j];
    }
    for (int i = 0; i < 4; i++) {
        U[i][0] =         t[i][0];
        U[i][1] = 0.5f * (t[i][0] + t[i][1] + t[i][2]);
        U[i][2] = 0.5f * (t[i][0] - t[i][1] + t[i][2]);
        U[i][3] =         t[i][2];
    }
}

/* A^T × M × A: 输出逆变换 4×4 → 2×2 */
static inline void transform_output(const float M[4][4], float out[2][2]) {
    float t[2][4];
    for (int j = 0; j < 4; j++) {
        t[0][j] = M[0][j] + M[1][j] + M[2][j];
        t[1][j] = M[1][j] - M[2][j] - M[3][j];
    }
    for (int i = 0; i < 2; i++) {
        out[i][0] = t[i][0] + t[i][1] + t[i][2];
        out[i][1] = t[i][1] - t[i][2] - t[i][3];
    }
}

/* ---- 权重变换缓存 (调用一次, 推理复用) ---- */
float* winograd_pack_weight(const float *weight, int OC, int C) {
    float *U = (float*)aligned_alloc(32, (size_t)ATILES * OC * C * sizeof(float));
    if (!U) return NULL;

    for (int a = 0; a < ATILES; a++) {
        int ai = a / ALPHA, aj = a % ALPHA;
        float *Ua = U + a * OC * C;
        for (int oc = 0; oc < OC; oc++) {
            for (int ic = 0; ic < C; ic++) {
                const float *gptr = weight + (oc * C + ic) * 9;
                float g[3][3], u[4][4];
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        g[i][j] = gptr[i*3+j];
                transform_weight(g, u);
                Ua[oc * C + ic] = u[ai][aj];
            }
        }
    }
    return U;
}

/*
 * winograd_conv_3x3 - Winograd F(2,3) 主函数
 * U_packed: 预变换权重 [16][OC*IC], 由 winograd_pack_weight 生成
 */
int winograd_conv_3x3(const float *X, int N, int C, int H, int W,
                      const float *U_packed, int OC,
                      const float *bias,
                      int PH, int PW,
                      float *Y, int OH, int OW,
                      int fused_relu) {
    int n_tile_h = (OH + TILE_M - 1) / TILE_M;
    int n_tile_w = (OW + TILE_M - 1) / TILE_M;
    int n_tiles  = n_tile_h * n_tile_w;
    int ohw = OH * OW;
    
    if (n_tiles < 4) return -1;

    size_t V_size = (size_t)ATILES * C * n_tiles;
    size_t M_size = (size_t)ATILES * OC * n_tiles;
    if (V_size + M_size > 64 * 1024 * 1024) return -1;

    float *V = (float*)malloc((V_size + M_size) * sizeof(float));
    if (!V) return -1;
    float *Mout = V + V_size;

    for (int n = 0; n < N; n++) {
        const float *Xn = X + n * C * H * W;
        float       *Yn = Y + n * OC * OH * OW;

        /* === Step 2: 输入变换 === */
        /* V[a][ic * n_tiles + tile_idx] = BT×d×B[ai][aj] */
        memset(V, 0, (size_t)ATILES * C * n_tiles * sizeof(float));

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(C * n_tiles >= 256)
        #endif
        for (int ic = 0; ic < C; ic++) {
            const float *Xc = Xn + ic * H * W;
            for (int ti = 0; ti < n_tile_h; ti++) {
                int oh0 = ti * TILE_M;
                for (int tj = 0; tj < n_tile_w; tj++) {
                    int ow0 = tj * TILE_M;
                    int tile_idx = ti * n_tile_w + tj;

                    /* 抽取 4×4 输入 tile (带 padding) */
                    float d[4][4];
                    for (int i = 0; i < 4; i++) {
                        int ih = oh0 + i - PH;
                        for (int j = 0; j < 4; j++) {
                            int iw = ow0 + j - PW;
                            d[i][j] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                      ? Xc[ih * W + iw] : 0.0f;
                        }
                    }

                    /* 变换并存入 V */
                    float Vt[4][4];
                    transform_input(d, Vt);
                    for (int a = 0; a < ATILES; a++) {
                        V[a * C * n_tiles + ic * n_tiles + tile_idx] =
                            Vt[a / ALPHA][a % ALPHA];
                    }
                }
            }
        }

        /* === Step 3: GEMM × 16 (变换域乘) ===
         * 每个 alpha 位置: M[a] = U[a] × V[a]
         * U[a]: OC×IC,  V[a]: IC×n_tiles → M[a]: OC×n_tiles
         */
        for (int a = 0; a < ATILES; a++) {
            sgemm_nn(OC, n_tiles, C,
                     U_packed + a * OC * C, C,
                     V + a * C * n_tiles, n_tiles,
                     Mout + a * OC * n_tiles, n_tiles,
                     NULL, 0, NULL);
        }

        /* === Step 4: 输出逆变换 + 写回 === */
        memset(Yn, 0, (size_t)OC * OH * OW * sizeof(float));

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static) if(OC * n_tiles >= 256)
        #endif
        for (int oc = 0; oc < OC; oc++) {
            float *Yoc = Yn + oc * ohw;
            float b = bias ? bias[oc] : 0.0f;

            for (int ti = 0; ti < n_tile_h; ti++) {
                int oh0 = ti * TILE_M;
                for (int tj = 0; tj < n_tile_w; tj++) {
                    int ow0 = tj * TILE_M;
                    int tile_idx = ti * n_tile_w + tj;

                    /* 收集 M[alpha][oc][tile_idx] → 4×4 矩阵 */
                    float M44[4][4];
                    for (int a = 0; a < ATILES; a++) {
                        M44[a/ALPHA][a%ALPHA] =
                            Mout[a * OC * n_tiles + oc * n_tiles + tile_idx];
                    }

                    float out2[2][2];
                    transform_output(M44, out2);

                    for (int i = 0; i < 2 && (oh0+i) < OH; i++) {
                        for (int j = 0; j < 2 && (ow0+j) < OW; j++) {
                            float val = out2[i][j] + b;
                            if (fused_relu && val < 0.0f) val = 0.0f;
                            Yoc[(oh0+i)*OW + (ow0+j)] = val;
                        }
                    }
                }
            }
        }
    }

    free(V);
    return 0;
}
