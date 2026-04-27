#include "../kernels.h"

/* ReduceMin: 最小值归约 */
int op_reduce_min(SpinnTensor **in, int n_in,
                  void *params, uint16_t params_size,
                  SpinnTensor **out, int n_out) {
    if (n_in < 1 || n_out < 1) return -1;
    
    float *X = (float*)in[0]->data;
    float *Y = (float*)out[0]->data;
    
    uint32_t n = in[0]->elem_count;
    float min_val = X[0];
    for (uint32_t i = 1; i < n; i++) {
        if (X[i] < min_val) min_val = X[i];
    }
    Y[0] = min_val;
    return 0;
}
