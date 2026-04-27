#include "../kernels.h"

/* Det: 矩阵行列式 (仅支持2x2和3x3) */
static float det_2x2(float *M) {
    return M[0] * M[3] - M[1] * M[2];
}

static float det_3x3(float *M) {
    return M[0] * (M[4] * M[8] - M[5] * M[7])
         - M[1] * (M[3] * M[8] - M[5] * M[6])
         + M[2] * (M[3] * M[7] - M[4] * M[6]);
}

int op_det(SpinnTensor **in, int n_in,
           void *params, uint16_t params_size,
           SpinnTensor **out, int n_out) {
    (void)params; (void)params_size;
    if (n_in < 1 || n_out < 1) return -1;
    
    float *X = (float*)in[0]->data;
    float *Y = (float*)out[0]->data;
    
    int n = in[0]->dims[in[0]->ndim - 1];
    int num_matrices = in[0]->elem_count / (n * n);
    
    for (int i = 0; i < num_matrices; i++) {
        float *M = X + i * n * n;
        
        if (n == 2) {
            Y[i] = det_2x2(M);
        } else if (n == 3) {
            Y[i] = det_3x3(M);
        } else {
            // 其他尺寸暂不支持
            Y[i] = 0.0f;
        }
    }
    
    return 0;
}
