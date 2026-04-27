#include "../kernels.h"

/* Softsign: y = x / (1 + |x|) */
int op_softsign(SpinnTensor **in, int n_in,
                void *params, uint16_t params_size,
                SpinnTensor **out, int n_out) {
    (void)params; (void)params_size;
    if (n_in < 1 || n_out < 1) return -1;
    
    float *X = (float*)in[0]->data;
    float *Y = (float*)out[0]->data;
    uint32_t n = in[0]->elem_count;
    
    for (uint32_t i = 0; i < n; i++) {
        float x = X[i];
        float abs_x = (x >= 0) ? x : -x;
        Y[i] = x / (1.0f + abs_x);
    }
    return 0;
}
