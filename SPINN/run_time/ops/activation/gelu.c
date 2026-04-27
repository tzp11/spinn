#include "../kernels.h"
#include <math.h>

/* Gelu: out = x * 0.5 * (1 + erf(x / sqrt(2))) */
int op_gelu(SpinnTensor **in, int n_in,
            void *params, uint16_t params_size,
            SpinnTensor **out, int n_out) {
    (void)params; (void)params_size;
    if (n_in < 1 || n_out < 1) return -1;
    
    float *x = (float*)in[0]->data;
    float *y = (float*)out[0]->data;
    uint32_t n = in[0]->elem_count;
    
    const float sqrt2_inv = 0.7071067811865475f;
    
    for (uint32_t i = 0; i < n; i++) {
        float v = x[i];
        y[i] = v * 0.5f * (1.0f + erff(v * sqrt2_inv));
    }
    return 0;
}
