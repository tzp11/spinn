#include "../kernels.h"
#include "simd_math.h"
#include <math.h>

/* Sigmoid: out = 1 / (1 + exp(-in)) */
int op_sigmoid(SpinnTensor **in, int n_in,
               void *params, uint16_t params_size,
               SpinnTensor **out, int n_out) {
    (void)params; (void)params_size;
    if (n_in < 1 || n_out < 1) return -1;

    float *x = (float*)in[0]->data;
    float *y = (float*)out[0]->data;
    uint32_t n = in[0]->elem_count;
    uint32_t i = 0;

#if SIMD_MATH_AVX2
    for (; i + 7 < n; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(y + i, spinn_sigmoid256_ps(v));
    }
#endif
    for (; i < n; i++) {
        y[i] = 1.0f / (1.0f + expf(-x[i]));
    }
    return 0;
}
