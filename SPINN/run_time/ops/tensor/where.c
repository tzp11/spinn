#include "../kernels.h"

/* Where: 条件选择 y = condition ? x : y */
int op_where(SpinnTensor **in, int n_in,
             void *params, uint16_t params_size,
             SpinnTensor **out, int n_out) {
    (void)params; (void)params_size;
    if (n_in < 3 || n_out < 1) return -1;
    
    // in[0]: condition (bool), in[1]: X, in[2]: Y
    uint8_t *condition = (uint8_t*)in[0]->data;
    float *X = (float*)in[1]->data;
    float *Y_in = (float*)in[2]->data;
    float *Y_out = (float*)out[0]->data;
    
    uint32_t n = out[0]->elem_count;
    uint32_t n_cond = in[0]->elem_count;
    uint32_t n_x = in[1]->elem_count;
    uint32_t n_y = in[2]->elem_count;
    
    for (uint32_t i = 0; i < n; i++) {
        Y_out[i] = condition[i % n_cond] ? X[i % n_x] : Y_in[i % n_y];
    }
    return 0;
}
