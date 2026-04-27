#include "../kernels.h"
#include "../../../ennf_op_params.h"
#include "gemm_kernel.h"

/* MatMul - 使用 Tiled GEMM 内核 */
int op_matmul(SpinnTensor **in, int n_in,
              void *params, uint16_t params_size,
              SpinnTensor **out, int n_out) {
    (void)params; (void)params_size;
    if (n_in < 2 || n_out < 1) return -1;
    
    float *A = (float*)in[0]->data;
    float *B = (float*)in[1]->data;
    float *C = (float*)out[0]->data;
    
    int a_ndim = in[0]->ndim;
    int b_ndim = in[1]->ndim;
    
    if (a_ndim == 2 && b_ndim == 2) {
        int M = in[0]->dims[0];
        int K = in[0]->dims[1];
        int N = in[1]->dims[1];
        sgemm_nn(M, N, K, A, K, B, N, C, N);
        return 0;
    }
    
    int M = in[0]->dims[a_ndim - 2];
    int K = in[0]->dims[a_ndim - 1];
    int K2 = in[1]->dims[b_ndim - 2];
    int N = in[1]->dims[b_ndim - 1];
    
    if (K != K2) {
        if (b_ndim == 2 && in[1]->dims[0] == (uint32_t)K) {
            int batch = 1;
            for (int i = 0; i < a_ndim - 2; i++) batch *= in[0]->dims[i];
            for (int b_idx = 0; b_idx < batch; b_idx++) {
                sgemm_nn(M, N, K,
                         A + b_idx * M * K, K,
                         B, N,
                         C + b_idx * M * N, N);
            }
            return 0;
        }
        return -1;
    }
    
    int batch = 1;
    for (int i = 0; i < a_ndim - 2; i++) batch *= in[0]->dims[i];
    int batch_b = 1;
    for (int i = 0; i < b_ndim - 2; i++) batch_b *= in[1]->dims[i];
    if (batch != batch_b) return -1;
    
    for (int b_idx = 0; b_idx < batch; b_idx++) {
        sgemm_nn(M, N, K,
                 A + b_idx * M * K, K,
                 B + b_idx * K * N, N,
                 C + b_idx * M * N, N);
    }
    return 0;
}
