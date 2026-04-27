#include "../kernels.h"
#include "../../../ennf_op_params.h"
#include "gemm_kernel.h"
#include <stdlib.h>

/* Gemm: Y = alpha * A @ B + beta * C  (使用 Tiled GEMM 内核) */
int op_gemm(SpinnTensor **in, int n_in,
            void *params, uint16_t params_size,
            SpinnTensor **out, int n_out) {
    if (n_in < 2 || n_out < 1) return -1;
    
    float *A = (float*)in[0]->data;
    float *B = (float*)in[1]->data;
    float *C_bias = (n_in > 2 && in[2]->data) ? (float*)in[2]->data : NULL;
    float *Y = (float*)out[0]->data;
    
    int transA = 0, transB = 0;
    float alpha = 1.0f, beta = 1.0f;
    
    if (params && params_size >= sizeof(ENNF_GemmParams)) {
        ENNF_GemmParams *p = (ENNF_GemmParams*)params;
        transA = p->transA;
        transB = p->transB;
        alpha = p->alpha;
        beta = p->beta;
    }
    
    int M = transA ? in[0]->dims[1] : in[0]->dims[0];
    int K = transA ? in[0]->dims[0] : in[0]->dims[1];
    int N = transB ? in[1]->dims[0] : in[1]->dims[1];
    int bias_is_vector = (C_bias && in[2]->elem_count == (uint32_t)N);
    
    /* 预处理 A: 如果 transA，转置到临时缓冲 */
    float *A_buf = NULL;
    const float *A_src = A;
    int lda = K;
    if (transA) {
        A_buf = (float*)malloc((size_t)M * K * sizeof(float));
        if (A_buf) {
            for (int m = 0; m < M; m++)
                for (int k = 0; k < K; k++)
                    A_buf[m * K + k] = A[k * M + m];
            A_src = A_buf;
        }
    }
    
    /* 预处理 B: 如果 transB，转置到临时缓冲 */
    float *B_buf = NULL;
    const float *B_src = B;
    int ldb = N;
    if (transB) {
        B_buf = (float*)malloc((size_t)K * N * sizeof(float));
        if (B_buf) {
            for (int k = 0; k < K; k++)
                for (int n = 0; n < N; n++)
                    B_buf[k * N + n] = B[n * K + k];
            B_src = B_buf;
        }
    }
    
    /* 预填充 bias 到 Y */
    if (C_bias && beta != 0.0f) {
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                int c_idx = bias_is_vector ? n : (m * N + n);
                Y[m * N + n] = beta * C_bias[c_idx];
            }
        /* GEMM: Y += alpha * A * B */
        sgemm_tiled(M, N, K, alpha, A_src, lda, B_src, ldb, 1.0f, Y, N);
    } else {
        /* GEMM: Y = alpha * A * B */
        sgemm_tiled(M, N, K, alpha, A_src, lda, B_src, ldb, 0.0f, Y, N);
    }
    
    if (A_buf) free(A_buf);
    if (B_buf) free(B_buf);
    return 0;
}
