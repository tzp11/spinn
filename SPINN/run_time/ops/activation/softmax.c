#include "../kernels.h"
#include "../../../ennf_op_params.h"
#include <math.h>
#include <string.h>

#if defined(__AVX2__) && defined(__FMA__)
#define USE_AVX2 1
#include <immintrin.h>
#endif

#if USE_AVX2
/*
 * 8 路 AVX2 expf 近似 (Cephes / sse_mathfun 标准实现, max abs error ~1 ULP).
 *   exp(x) = 2^k * exp(z),  k = round(x*log2(e)),  z = x - k*ln(2)
 *   ln(2) 用高低位分解 (LN2_HI + LN2_LO), 减少抵消误差
 *   exp(z) 在 z ∈ [-ln2/2, ln2/2] 上用 7 阶多项式
 */
static inline __m256 exp256_ps(__m256 x) {
    const __m256 ONE     = _mm256_set1_ps(1.0f);
    const __m256 HALF    = _mm256_set1_ps(0.5f);
    const __m256 EXP_HI  = _mm256_set1_ps( 88.3762626647949f);
    const __m256 EXP_LO  = _mm256_set1_ps(-88.3762626647949f);
    const __m256 LOG2EF  = _mm256_set1_ps(1.44269504088896341f);
    const __m256 LN2_HI  = _mm256_set1_ps(0.693359375f);
    const __m256 LN2_LO  = _mm256_set1_ps(-2.12194440e-4f);

    x = _mm256_min_ps(x, EXP_HI);
    x = _mm256_max_ps(x, EXP_LO);

    /* fx = floor(x * log2(e) + 0.5) */
    __m256 fx = _mm256_fmadd_ps(x, LOG2EF, HALF);
    fx = _mm256_floor_ps(fx);

    /* z = x - fx * ln2 (高低位减少抵消) */
    __m256 z = _mm256_fnmadd_ps(fx, LN2_HI, x);
    z = _mm256_fnmadd_ps(fx, LN2_LO, z);

    /* exp(z) 7 阶多项式 (Cephes) */
    __m256 z2 = _mm256_mul_ps(z, z);
    __m256 p = _mm256_set1_ps(1.9875691500e-4f);
    p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(1.3981999507e-3f));
    p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(8.3334519073e-3f));
    p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(4.1665795894e-2f));
    p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(1.6666665459e-1f));
    p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(5.0000001201e-1f));
    p = _mm256_fmadd_ps(p, z2, z);
    p = _mm256_add_ps(p, ONE);

    /* 2^fx via IEEE bit-pattern */
    __m256i ki = _mm256_cvtps_epi32(fx);
    __m256i bias = _mm256_set1_epi32(127);
    __m256 pow2 = _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_add_epi32(ki, bias), 23));

    return _mm256_mul_ps(p, pow2);
}

/* 横向归约: __m256 -> max scalar */
static inline float hmax256(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 m = _mm_max_ps(lo, hi);
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
    return _mm_cvtss_f32(m);
}

/* 横向归约: __m256 -> sum scalar */
static inline float hsum256(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

/* inner=1 的 SIMD softmax: 一行 dim 个连续元素 */
static void softmax_row_avx2(const float *x, float *y, int dim) {
    /* Pass 1: max */
    int j = 0;
    __m256 vmax = _mm256_set1_ps(-INFINITY);
    for (; j + 7 < dim; j += 8) {
        vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(x + j));
    }
    float max_val = (j > 0) ? hmax256(vmax) : x[0];
    for (; j < dim; j++) if (x[j] > max_val) max_val = x[j];

    /* Pass 2: exp(x - max) + sum */
    __m256 vmaxb = _mm256_set1_ps(max_val);
    __m256 vsum = _mm256_setzero_ps();
    j = 0;
    for (; j + 7 < dim; j += 8) {
        __m256 v = _mm256_sub_ps(_mm256_loadu_ps(x + j), vmaxb);
        v = exp256_ps(v);
        _mm256_storeu_ps(y + j, v);
        vsum = _mm256_add_ps(vsum, v);
    }
    float sum = (j > 0) ? hsum256(vsum) : 0.0f;
    for (; j < dim; j++) {
        float v = expf(x[j] - max_val);
        y[j] = v;
        sum += v;
    }

    /* Pass 3: scale */
    if (sum == 0.0f) sum = 1e-10f;
    float inv = 1.0f / sum;
    __m256 vinv = _mm256_set1_ps(inv);
    j = 0;
    for (; j + 7 < dim; j += 8) {
        _mm256_storeu_ps(y + j, _mm256_mul_ps(_mm256_loadu_ps(y + j), vinv));
    }
    for (; j < dim; j++) y[j] *= inv;
}
#endif /* USE_AVX2 */

/* Softmax: out = exp(in) / sum(exp(in)) along axis */
int op_softmax(SpinnTensor **in, int n_in,
               void *params, uint16_t params_size,
               SpinnTensor **out, int n_out) {
    if (n_in < 1 || n_out < 1) return -1;

    int axis = -1;
    if (params && params_size >= sizeof(ENNF_SoftmaxParams)) {
        ENNF_SoftmaxParams *p = (ENNF_SoftmaxParams*)params;
        axis = p->axis;
    }

    int ndim = in[0]->ndim;
    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim) return -1;

    int outer = 1;
    for (int i = 0; i < axis; i++) outer *= in[0]->dims[i];
    int dim = in[0]->dims[axis];
    int inner = 1;
    for (int i = axis + 1; i < ndim; i++) inner *= in[0]->dims[i];

    float *x = (float*)in[0]->data;
    float *y = (float*)out[0]->data;

#if USE_AVX2
    /* inner=1 fast path (axis 是最后一维, 内存连续): SIMD 化 */
    if (inner == 1 && dim >= 8) {
        for (int i = 0; i < outer; i++) {
            softmax_row_avx2(x + i * dim, y + i * dim, dim);
        }
        return 0;
    }
#endif

    /* 通用 fallback: 标量 (inner > 1 是 stride 访问, 暂不优化) */
    for (int i = 0; i < outer; i++) {
        for (int k = 0; k < inner; k++) {
            int base_ptr = i * dim * inner + k;
            float max_val = x[base_ptr];
            for (int j = 1; j < dim; j++) {
                float val = x[base_ptr + j * inner];
                if (val > max_val) max_val = val;
            }
            float sum = 0;
            for (int j = 0; j < dim; j++) {
                float v = expf(x[base_ptr + j * inner] - max_val);
                y[base_ptr + j * inner] = v;
                sum += v;
            }
            if (sum == 0) sum = 1e-10f;
            float scale = 1.0f / sum;
            for (int j = 0; j < dim; j++) {
                y[base_ptr + j * inner] *= scale;
            }
        }
    }
    return 0;
}
