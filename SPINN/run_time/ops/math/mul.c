#include "../kernels.h"
#include <math.h>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define USE_AVX2 1
#else
#define USE_AVX2 0
#endif

/* Mul: out = in[0] * in[1] */
int op_mul(SpinnTensor **in, int n_in,
           void *params, uint16_t params_size,
           SpinnTensor **out, int n_out) {
    (void)params; (void)params_size;
    if (n_in < 2 || n_out < 1) return -1;
    
    float *a = (float*)in[0]->data;
    float *b = (float*)in[1]->data;
    float *y = (float*)out[0]->data;
    
    uint32_t n = out[0]->elem_count;
    uint32_t n_a = in[0]->elem_count;
    uint32_t n_b = in[1]->elem_count;
    
    /* 快速路径: 相同大小的逐元素乘法 */
    if (n_a == n_b && n_a == n) {
        uint32_t i = 0;
#if USE_AVX2
        for (; i + 7 < n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 vy = _mm256_mul_ps(va, vb);
            _mm256_storeu_ps(y + i, vy);
        }
#endif
        for (; i < n; i++) {
            y[i] = a[i] * b[i];
        }
    } else {
        /* 广播路径: 常见模式 a[1] * b[N] 或 a[N] * b[1] */
        if (n_a == 1 || n_b == 1) {
            const float *vec = (n_a == 1) ? b : a;
            float scalar = (n_a == 1) ? a[0] : b[0];
            uint32_t i = 0;
#if USE_AVX2
            __m256 vs = _mm256_set1_ps(scalar);
            for (; i + 7 < n; i += 8) {
                __m256 vv = _mm256_loadu_ps(vec + i);
                _mm256_storeu_ps(y + i, _mm256_mul_ps(vs, vv));
            }
#endif
            for (; i < n; i++) {
                y[i] = scalar * vec[i];
            }
        } else {
            /* 通用广播 */
            for (uint32_t i = 0; i < n; i++) {
                y[i] = a[i % n_a] * b[i % n_b];
            }
        }
    }
    return 0;
}
