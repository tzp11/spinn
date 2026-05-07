/* Minimal Winograd F(2,3) step-by-step test */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define ALPHA 4
#define TILE_M 2
#define ATILES 16

/* Same transforms as winograd.c */
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

int main() {
    /* Tiny test: C=1, H=4, W=4, OC=1, KH=KW=3, SH=SW=1, PH=PW=1 */
    int C = 1, H = 4, W = 4, OC = 1;
    int OH = H, OW = W;

    float X[16] = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
    float weight[9] = {1,0,0, 0,1,0, 0,0,1}; /* identity kernel */
    float bias_val[1] = {0};

    /* Direct conv: identity 3x3 with PH=PW=1 → output = input */
    printf("Direct conv output (should equal input for identity kernel):\n");
    for (int oh = 0; oh < OH; oh++) {
        for (int ow = 0; ow < OW; ow++) {
            float s = 0;
            for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                    int ih = oh - 1 + kh, iw = ow - 1 + kw;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                        s += X[ih * W + iw] * weight[kh * 3 + kw];
                }
            }
            printf("%.1f ", s);
        }
        printf("\n");
    }

    /* Winograd: single OC, single IC, 2x2 tiles */
    int n_tile_h = (OH + 1) / 2;  /* = 2 */
    int n_tile_w = (OW + 1) / 2;  /* = 2 */
    int n_tiles = n_tile_h * n_tile_w;  /* = 4 */

    /* Weight transform */
    float g[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
    float U[4][4];
    transform_weight(g, U);
    printf("\nU (weight transform):\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) printf("%.3f ", U[i][j]);
        printf("\n");
    }

    /* Input transform for each tile */
    float V[4][4];  /* only 1 tile at a time for debugging */
    float Y_win[OH * OW];
    memset(Y_win, 0, sizeof(Y_win));

    for (int ti = 0; ti < n_tile_h; ti++) {
        for (int tj = 0; tj < n_tile_w; tj++) {
            int oh0 = ti * 2, ow0 = tj * 2;

            /* Extract 4x4 input tile */
            float d[4][4];
            for (int i = 0; i < 4; i++) {
                int ih = oh0 + i - 1;  /* PH=1 */
                for (int j = 0; j < 4; j++) {
                    int iw = ow0 + j - 1;
                    d[i][j] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                              ? X[ih * W + iw] : 0.0f;
                }
            }
            printf("\nTile (%d,%d) input:\n", ti, tj);
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) printf("%.1f ", d[i][j]);
                printf("\n");
            }

            /* Input transform */
            transform_input(d, V);
            printf("V (input transform):\n");
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) printf("%.3f ", V[i][j]);
                printf("\n");
            }

            /* Element-wise multiply: M[a][b] = U[a][b] * V[a][b] */
            float M[4][4];
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    M[i][j] = U[i][j] * V[i][j];
            printf("M (element-wise product):\n");
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) printf("%.3f ", M[i][j]);
                printf("\n");
            }

            /* Output transform */
            float out[2][2];
            transform_output(M, out);
            printf("Output tile: %.2f %.2f / %.2f %.2f\n",
                   out[0][0], out[0][1], out[1][0], out[1][1]);

            /* Write back */
            for (int i = 0; i < 2 && oh0+i < OH; i++)
                for (int j = 0; j < 2 && ow0+j < OW; j++)
                    Y_win[(oh0+i)*OW + ow0+j] = out[i][j];
        }
    }

    printf("\nWinograd output:\n");
    for (int oh = 0; oh < OH; oh++) {
        for (int ow = 0; ow < OW; ow++) printf("%.1f ", Y_win[oh*OW+ow]);
        printf("\n");
    }

    /* Compare */
    float max_diff = 0;
    for (int oh = 0; oh < OH; oh++) {
        for (int ow = 0; ow < OW; ow++) {
            float direct = 0;
            for (int kh = 0; kh < 3; kh++) {
                for (int kw = 0; kw < 3; kw++) {
                    int ih = oh - 1 + kh, iw = ow - 1 + kw;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                        direct += X[ih * W + iw] * weight[kh * 3 + kw];
                }
            }
            float diff = fabsf(direct - Y_win[oh*OW+ow]);
            if (diff > max_diff) max_diff = diff;
        }
    }
    printf("\nMax diff: %e\n", max_diff);
    return max_diff > 0.01f ? 1 : 0;
}
