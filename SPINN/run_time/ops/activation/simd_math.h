/*
 * simd_math.h - 共享 SIMD 数学近似函数
 *
 * 提供 AVX2 向量化的 expf / sigmoid 等近似实现, 供 softmax / sigmoid /
 * Conv+SiLU 等算子复用. 精度 ~1 ULP, 与 libm expf 数值等价.
 */
#ifndef __SPINN_SIMD_MATH_H__
#define __SPINN_SIMD_MATH_H__

#if defined(__AVX2__) && defined(__FMA__)
#define SIMD_MATH_AVX2 1
#include <immintrin.h>

/*
 * 8 路 AVX2 expf (Cephes / sse_mathfun 标准实现, max abs error ~1 ULP).
 *   exp(x) = 2^k * exp(z),  k = round(x*log2(e)),  z = x - k*ln(2)
 *   ln(2) 用高低位分解减少抵消误差, exp(z) 用 7 阶多项式
 */
static inline __m256 spinn_exp256_ps(__m256 x) {
    const __m256 ONE     = _mm256_set1_ps(1.0f);
    const __m256 HALF    = _mm256_set1_ps(0.5f);
    const __m256 EXP_HI  = _mm256_set1_ps( 88.3762626647949f);
    const __m256 EXP_LO  = _mm256_set1_ps(-88.3762626647949f);
    const __m256 LOG2EF  = _mm256_set1_ps(1.44269504088896341f);
    const __m256 LN2_HI  = _mm256_set1_ps(0.693359375f);
    const __m256 LN2_LO  = _mm256_set1_ps(-2.12194440e-4f);

    x = _mm256_min_ps(x, EXP_HI);
    x = _mm256_max_ps(x, EXP_LO);

    __m256 fx = _mm256_fmadd_ps(x, LOG2EF, HALF);
    fx = _mm256_floor_ps(fx);

    __m256 z = _mm256_fnmadd_ps(fx, LN2_HI, x);
    z = _mm256_fnmadd_ps(fx, LN2_LO, z);

    __m256 z2 = _mm256_mul_ps(z, z);
    __m256 p = _mm256_set1_ps(1.9875691500e-4f);
    p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(1.3981999507e-3f));
    p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(8.3334519073e-3f));
    p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(4.1665795894e-2f));
    p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(1.6666665459e-1f));
    p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(5.0000001201e-1f));
    p = _mm256_fmadd_ps(p, z2, z);
    p = _mm256_add_ps(p, ONE);

    __m256i ki = _mm256_cvtps_epi32(fx);
    __m256i bias = _mm256_set1_epi32(127);
    __m256 pow2 = _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_add_epi32(ki, bias), 23));

    return _mm256_mul_ps(p, pow2);
}

/* sigmoid(x) = 1 / (1 + exp(-x)) */
static inline __m256 spinn_sigmoid256_ps(__m256 x) {
    const __m256 ONE = _mm256_set1_ps(1.0f);
    __m256 neg = _mm256_sub_ps(_mm256_setzero_ps(), x);
    __m256 e = spinn_exp256_ps(neg);
    return _mm256_div_ps(ONE, _mm256_add_ps(ONE, e));
}

/* 横向归约: __m256 -> max scalar */
static inline float spinn_hmax256(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 m = _mm_max_ps(lo, hi);
    m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
    return _mm_cvtss_f32(m);
}

/* 横向归约: __m256 -> sum scalar */
static inline float spinn_hsum256(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

#endif /* AVX2 */

#endif /* __SPINN_SIMD_MATH_H__ */
