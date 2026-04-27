#include "../kernels.h"

/* ReduceMax: 最大值归约 */
int op_reduce_max(SpinnTensor **in, int n_in,
                  void *params, uint16_t params_size,
                  SpinnTensor **out, int n_out) {
    if (n_in < 1 || n_out < 1) return -1;
    
    float *X = (float*)in[0]->data;
    float *Y = (float*)out[0]->data;
    
    uint32_t out_n = out[0]->elem_count;
    uint32_t in_n = in[0]->elem_count;
    
    if (out_n == 0) return 0;
    uint32_t block_size = in_n / out_n;
    
    for (uint32_t i = 0; i < out_n; i++) {
        // Initialize with first element of the block
        float max_val = X[i * block_size];
        for (uint32_t j = 1; j < block_size; j++) {
            float v = X[i * block_size + j];
            if (v > max_val) max_val = v;
        }
        Y[i] = max_val;
    }
    return 0;
}
