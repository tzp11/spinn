#include "../kernels.h"
#include <math.h>

/* Softplus: y = log(exp(x) + 1) */
int op_softplus(SpinnTensor **in, int n_in,
                void *params, uint16_t params_size,
                SpinnTensor **out, int n_out) {
    (void)params; (void)params_size;
    if (n_in < 1 || n_out < 1) return -1;
    
    float *X = (float*)in[0]->data;
    float *Y = (float*)out[0]->data;
    uint32_t n = in[0]->elem_count;
    
    for (uint32_t i = 0; i < n; i++) {
        // 数值稳定版本
        float x = X[i];
        if (x > 20.0f) {
            Y[i] = x; // exp(x) >> 1
        } else if (x < -20.0f) {
            Y[i] = expf(x); // exp(x) + 1 ≈ 1
        } else {
            Y[i] = logf(1.0f + expf(x));
        }
    }
    return 0;
}
