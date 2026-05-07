#include "../kernels.h"

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define USE_AVX2 1
#else
#define USE_AVX2 0
#endif

/* HardSwish: out = x * max(0, min(1, (x + 3) / 6)) */
int op_hard_swish(SpinnTensor **in, int n_in,
                  void *params, uint16_t params_size,
                  SpinnTensor **out, int n_out) {
    (void)params; (void)params_size;
    if (n_in < 1 || n_out < 1) return -1;
    
    float *x = (float*)in[0]->data;
    float *y = (float*)out[0]->data;
    uint32_t n = in[0]->elem_count;
    uint32_t i = 0;
    
#if USE_AVX2
    const __m256 v3 = _mm256_set1_ps(3.0f);
    const __m256 v1_6 = _mm256_set1_ps(1.0f / 6.0f);
    const __m256 v0 = _mm256_setzero_ps();
    const __m256 v1 = _mm256_set1_ps(1.0f);
    
    for (; i + 7 < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vh = _mm256_mul_ps(_mm256_add_ps(vx, v3), v1_6);
        vh = _mm256_max_ps(v0, _mm256_min_ps(v1, vh));
        __m256 vy = _mm256_mul_ps(vx, vh);
        _mm256_storeu_ps(y + i, vy);
    }
#endif
    
    for (; i < n; i++) {
        float v = x[i];
        float h = (v + 3.0f) / 6.0f;
        if (h < 0) h = 0;
        if (h > 1) h = 1;
        y[i] = v * h;
    }
    return 0;
}
