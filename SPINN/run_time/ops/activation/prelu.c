#include "../kernels.h"

/* PRelu: out = x if x > 0 else slope * x */
int op_prelu(SpinnTensor **in, int n_in,
             void *params, uint16_t params_size,
             SpinnTensor **out, int n_out) {
    (void)params; (void)params_size;
    if (n_in < 2 || n_out < 1) return -1;
    
    float *x = (float*)in[0]->data;
    float *slope = (float*)in[1]->data;
    float *y = (float*)out[0]->data;
    
    uint32_t n = in[0]->elem_count;
    uint32_t n_slope = in[1]->elem_count;
    
    for (uint32_t i = 0; i < n; i++) {
        float v = x[i];
        float s = slope[i % n_slope];
        y[i] = (v > 0) ? v : s * v;
    }
    return 0;
}
