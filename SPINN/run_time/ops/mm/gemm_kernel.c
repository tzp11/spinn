/*
 * gemm_kernel.c - 高性能 Tiled GEMM (最终优化版)
 *
 * 参照 ORT MLAS sgemm.cpp 的关键实现：
 *   1. 6×16 寄存器驻留微内核 (15/16 YMM 满占)
 *   2. k-major B 打包 + AVX2 向量化 pack
 *   3. OpenMP 智能阈值 (小 GEMM 跳过多线程)
 *   4. 栈上 B panel (零 malloc)
 *   5. 4K 展开 K 循环
 *
 * 寄存器分配 (6×16, i7-9700K 16 YMM):
 *   c0L,c0R ... c5L,c5R = 6行×2 = 12 个 (C 累加)
 *   bL, bR               = 2 个 (B 当前行)
 *   va                   = 1 个 (A 广播)
 *   Total: 15/16 — 仅留 1 给编译器
 */

#include "gemm_kernel.h"
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__AVX2__) && defined(__FMA__)
  #define USE_AVX2 1
  #include <immintrin.h>
#endif

#define TILE_MR  6     /* 微内核行数 (ORT 用 6) */
#define TILE_NR  16    /* 微内核列数 (2 × AVX宽度) */
#define TILE_KC  256   /* K 分块 */
#define TILE_NC  128   /* N 分块 */

/* 小矩阵不启用 OpenMP 的阈值 */
#define OMP_MIN_M  24

/* ============================================================
 * 线程局部 B-pack 缓冲: 消除 sgemm_nn_packed_a 每次调用的 malloc 开销.
 * ResNet101 的 832 次 Conv 调用 -> 832 次 malloc/free 全部消除.
 * 容量按需增长, 程序结束由 OS 回收 (thread_local 不需主动 free).
 * ============================================================ */
static _Thread_local float *tls_b_pack = NULL;
static _Thread_local size_t tls_b_pack_floats = 0;

static float* get_b_pack_buf(size_t need_floats) {
    if (tls_b_pack_floats < need_floats) {
        free(tls_b_pack);
        size_t bytes = need_floats * sizeof(float);
        if (posix_memalign((void**)&tls_b_pack, 32, bytes) != 0) {
            tls_b_pack = NULL;
            tls_b_pack_floats = 0;
            return NULL;
        }
        tls_b_pack_floats = need_floats;
    }
    return tls_b_pack;
}

/* ============================================================
 * B 打包: k-major 行优先, AVX2 向量化
 * pb[k * TILE_NR + n]
 * ============================================================ */
static inline void pack_b(float * __restrict__ pb,
                           const float * __restrict__ B, int ldb,
                           int ck, int cn) {
    for (int k = 0; k < ck; k++) {
        const float *src = B + k * ldb;
        float *dst = pb + k * TILE_NR;
        int n = 0;
#if USE_AVX2
        for (; n + 7 < cn; n += 8)
            _mm256_storeu_ps(dst + n, _mm256_loadu_ps(src + n));
#endif
        for (; n < cn; n++) dst[n] = src[n];
        for (; n < TILE_NR; n++) dst[n] = 0.0f;
    }
}

#if USE_AVX2

/* ============================================================
 * 6×16 AVX2 FMA 寄存器驻留微内核
 *
 * 15 个 YMM 寄存器全部占用:
 *   c0L..c5L, c0R..c5R (12) + bL, bR (2) + va (1) = 15
 * ============================================================ */

/* K 循环一步的宏：加载 B，广播 A 的 6 行，12 次 FMA */
#define KERNEL_STEP(KK) \
    bL = _mm256_loadu_ps(pb + (KK) * TILE_NR); \
    bR = _mm256_loadu_ps(pb + (KK) * TILE_NR + 8); \
    va = _mm256_set1_ps(a0[KK]); c0L = _mm256_fmadd_ps(va, bL, c0L); c0R = _mm256_fmadd_ps(va, bR, c0R); \
    va = _mm256_set1_ps(a1[KK]); c1L = _mm256_fmadd_ps(va, bL, c1L); c1R = _mm256_fmadd_ps(va, bR, c1R); \
    va = _mm256_set1_ps(a2[KK]); c2L = _mm256_fmadd_ps(va, bL, c2L); c2R = _mm256_fmadd_ps(va, bR, c2R); \
    va = _mm256_set1_ps(a3[KK]); c3L = _mm256_fmadd_ps(va, bL, c3L); c3R = _mm256_fmadd_ps(va, bR, c3R); \
    va = _mm256_set1_ps(a4[KK]); c4L = _mm256_fmadd_ps(va, bL, c4L); c4R = _mm256_fmadd_ps(va, bR, c4R); \
    va = _mm256_set1_ps(a5[KK]); c5L = _mm256_fmadd_ps(va, bL, c5L); c5R = _mm256_fmadd_ps(va, bR, c5R);

static void micro_6x16(const float * __restrict__ A, int lda,
                        const float * __restrict__ pb,
                        float * __restrict__ C, int ldc,
                        int ck, int actual_m, int actual_n,
                        int zero_mode) {
    /* 边界回退 */
    if (actual_m < TILE_MR || actual_n < TILE_NR) {
        for (int m = 0; m < actual_m; m++) {
            const float *ar = A + m * lda;
            for (int n = 0; n < actual_n; n++) {
                __m256 vs0 = _mm256_setzero_ps();
                __m256 vs1 = _mm256_setzero_ps();
                int k = 0;
                for (; k + 15 < ck; k += 16) {
                    vs0 = _mm256_fmadd_ps(_mm256_loadu_ps(ar+k),
                          _mm256_set_ps(pb[(k+7)*TILE_NR+n],pb[(k+6)*TILE_NR+n],
                                        pb[(k+5)*TILE_NR+n],pb[(k+4)*TILE_NR+n],
                                        pb[(k+3)*TILE_NR+n],pb[(k+2)*TILE_NR+n],
                                        pb[(k+1)*TILE_NR+n],pb[(k+0)*TILE_NR+n]), vs0);
                    vs1 = _mm256_fmadd_ps(_mm256_loadu_ps(ar+k+8),
                          _mm256_set_ps(pb[(k+15)*TILE_NR+n],pb[(k+14)*TILE_NR+n],
                                        pb[(k+13)*TILE_NR+n],pb[(k+12)*TILE_NR+n],
                                        pb[(k+11)*TILE_NR+n],pb[(k+10)*TILE_NR+n],
                                        pb[(k+9)*TILE_NR+n], pb[(k+8)*TILE_NR+n]), vs1);
                }
                vs0 = _mm256_add_ps(vs0, vs1);
                __m128 hi = _mm256_extractf128_ps(vs0, 1);
                __m128 lo = _mm256_castps256_ps128(vs0);
                __m128 s4 = _mm_add_ps(lo, hi);
                s4 = _mm_hadd_ps(s4, s4);
                s4 = _mm_hadd_ps(s4, s4);
                float sum = _mm_cvtss_f32(s4);
                for (; k < ck; k++) sum += ar[k] * pb[k * TILE_NR + n];
                if (zero_mode) C[m * ldc + n] = sum;
                else           C[m * ldc + n] += sum;
            }
        }
        return;
    }

    /* ========== 主路径: 6×16 寄存器驻留 ========== */
    __m256 c0L, c0R, c1L, c1R, c2L, c2R, c3L, c3R, c4L, c4R, c5L, c5R;
    __m256 bL, bR, va;

    if (zero_mode) {
        c0L=_mm256_setzero_ps(); c0R=_mm256_setzero_ps();
        c1L=_mm256_setzero_ps(); c1R=_mm256_setzero_ps();
        c2L=_mm256_setzero_ps(); c2R=_mm256_setzero_ps();
        c3L=_mm256_setzero_ps(); c3R=_mm256_setzero_ps();
        c4L=_mm256_setzero_ps(); c4R=_mm256_setzero_ps();
        c5L=_mm256_setzero_ps(); c5R=_mm256_setzero_ps();
    } else {
        c0L=_mm256_loadu_ps(C);            c0R=_mm256_loadu_ps(C+8);
        c1L=_mm256_loadu_ps(C+ldc);        c1R=_mm256_loadu_ps(C+ldc+8);
        c2L=_mm256_loadu_ps(C+2*ldc);      c2R=_mm256_loadu_ps(C+2*ldc+8);
        c3L=_mm256_loadu_ps(C+3*ldc);      c3R=_mm256_loadu_ps(C+3*ldc+8);
        c4L=_mm256_loadu_ps(C+4*ldc);      c4R=_mm256_loadu_ps(C+4*ldc+8);
        c5L=_mm256_loadu_ps(C+5*ldc);      c5R=_mm256_loadu_ps(C+5*ldc+8);
    }

    const float *a0=A, *a1=A+lda, *a2=A+2*lda, *a3=A+3*lda, *a4=A+4*lda, *a5=A+5*lda;

    int k = 0;
    for (; k + 3 < ck; k += 4) {
        KERNEL_STEP(k+0)
        KERNEL_STEP(k+1)
        KERNEL_STEP(k+2)
        KERNEL_STEP(k+3)
    }
    for (; k < ck; k++) {
        KERNEL_STEP(k)
    }

    _mm256_storeu_ps(C,         c0L); _mm256_storeu_ps(C+8,         c0R);
    _mm256_storeu_ps(C+ldc,     c1L); _mm256_storeu_ps(C+ldc+8,     c1R);
    _mm256_storeu_ps(C+2*ldc,   c2L); _mm256_storeu_ps(C+2*ldc+8,   c2R);
    _mm256_storeu_ps(C+3*ldc,   c3L); _mm256_storeu_ps(C+3*ldc+8,   c3R);
    _mm256_storeu_ps(C+4*ldc,   c4L); _mm256_storeu_ps(C+4*ldc+8,   c4R);
    _mm256_storeu_ps(C+5*ldc,   c5L); _mm256_storeu_ps(C+5*ldc+8,   c5R);
}

#undef KERNEL_STEP

#endif /* USE_AVX2 */

/* 标量回退 */
static void __attribute__((unused)) micro_scalar(const float * __restrict__ A, int lda,
                          const float * __restrict__ pb,
                          float * __restrict__ C, int ldc,
                          int ck, int cm, int cn, int zero_mode) {
    for (int m = 0; m < cm; m++) {
        const float *ar = A + m * lda;
        for (int n = 0; n < cn; n++) {
            float sum = 0.0f;
            for (int k = 0; k < ck; k++) sum += ar[k] * pb[k*TILE_NR+n];
            if (zero_mode) C[m*ldc+n] = sum;
            else           C[m*ldc+n] += sum;
        }
    }
}


/* ============================================================
 * 主函数
 * ============================================================ */
void sgemm_tiled(int M, int N, int K,
                 float alpha,
                 const float *A, int lda,
                 const float *B, int ldb,
                 float beta,
                 float *C, int ldc) {
    if (M <= 0 || N <= 0 || K <= 0) return;

    /* 小矩阵快速通道: M=1 (全连接层等) */
    if (M == 1) {
        if (beta == 0.0f) memset(C, 0, N * sizeof(float));
        else if (beta != 1.0f) for (int n = 0; n < N; n++) C[n] *= beta;
        
        for (int k = 0; k < K; k++) {
            float a_val = alpha * A[k];
            const float *b_row = B + k * ldb;
            int n = 0;
#if USE_AVX2
            __m256 va = _mm256_set1_ps(a_val);
            for (; n + 31 < N; n += 32) {
                __m256 c0 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b_row + n), _mm256_loadu_ps(C + n));
                _mm256_storeu_ps(C + n, c0);
                __m256 c1 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b_row + n + 8), _mm256_loadu_ps(C + n + 8));
                _mm256_storeu_ps(C + n + 8, c1);
                __m256 c2 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b_row + n + 16), _mm256_loadu_ps(C + n + 16));
                _mm256_storeu_ps(C + n + 16, c2);
                __m256 c3 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b_row + n + 24), _mm256_loadu_ps(C + n + 24));
                _mm256_storeu_ps(C + n + 24, c3);
            }
#endif
            for (; n < N; n++) {
                C[n] += a_val * b_row[n];
            }
        }
        return;
    }

    /* beta 处理 */
    if (beta == 0.0f) {
        for (int m = 0; m < M; m++) memset(C + m*ldc, 0, N*sizeof(float));
    } else if (beta != 1.0f) {
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) C[m*ldc+n] *= beta;
    }

    int need_scale = (alpha != 1.0f);
    int use_par = 0;
#ifdef _OPENMP
    use_par = (M >= OMP_MIN_M);
#endif

    for (int n0 = 0; n0 < N; n0 += TILE_NC) {
        int cn = (N-n0 < TILE_NC) ? (N-n0) : TILE_NC;

        for (int k0 = 0; k0 < K; k0 += TILE_KC) {
            int ck = (K-k0 < TILE_KC) ? (K-k0) : TILE_KC;
            int zm = (beta == 0.0f && k0 == 0) ? 1 : 0;
            if (beta != 0.0f) zm = 0;

            for (int nj = 0; nj < cn; nj += TILE_NR) {
                int cnr = (cn-nj < TILE_NR) ? (cn-nj) : TILE_NR;

                float pb[TILE_KC * TILE_NR] __attribute__((aligned(32)));
                pack_b(pb, B+k0*ldb+n0+nj, ldb, ck, cnr);

                #ifdef _OPENMP
                if (use_par) {
                    #pragma omp parallel for schedule(static)
                    for (int m0 = 0; m0 < M; m0 += TILE_MR) {
                        int cm = (M-m0 < TILE_MR) ? (M-m0) : TILE_MR;
                        #if USE_AVX2
                        micro_6x16(A+m0*lda+k0, lda, pb, C+m0*ldc+n0+nj, ldc, ck, cm, cnr, zm);
                        #else
                        micro_scalar(A+m0*lda+k0, lda, pb, C+m0*ldc+n0+nj, ldc, ck, cm, cnr, zm);
                        #endif
                    }
                } else
                #endif
                {
                    for (int m0 = 0; m0 < M; m0 += TILE_MR) {
                        int cm = (M-m0 < TILE_MR) ? (M-m0) : TILE_MR;
                        #if USE_AVX2
                        micro_6x16(A+m0*lda+k0, lda, pb, C+m0*ldc+n0+nj, ldc, ck, cm, cnr, zm);
                        #else
                        micro_scalar(A+m0*lda+k0, lda, pb, C+m0*ldc+n0+nj, ldc, ck, cm, cnr, zm);
                        #endif
                    }
                }
            }
        }
    }

    if (need_scale) {
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) C[m*ldc+n] *= alpha;
    }
}

void sgemm_nn(int M, int N, int K,
              const float *A, int lda,
              const float *B, int ldb,
              float *C, int ldc) {
    sgemm_tiled(M, N, K, 1.0f, A, lda, B, ldb, 0.0f, C, ldc);
}

/* ============================================================
 * 静态权重预打包 (Offline Pack A)
 * 内存布局: (M // TILE_MR) × K × TILE_MR，最后一块补齐零。
 * 这样微内核能纯按列顺序连续读取！
 * ============================================================ */
void* sgemm_pack_a_offline(int M, int K, const float *A, int lda, size_t *out_size) {
    int num_m_blocks = (M + TILE_MR - 1) / TILE_MR;
    size_t total_floats = (size_t)num_m_blocks * K * TILE_MR;
    *out_size = total_floats * sizeof(float);
    
    void *packed_data = NULL;
    if (posix_memalign(&packed_data, 32, *out_size) != 0) return NULL;
    float *pa = (float*)packed_data;
    
    for (int m0 = 0; m0 < M; m0 += TILE_MR) {
        int cm = (M - m0 < TILE_MR) ? (M - m0) : TILE_MR;
        for (int k = 0; k < K; k++) {
            int m = 0;
            for (; m < cm; m++) {
                pa[m] = A[(m0 + m) * lda + k];
            }
            for (; m < TILE_MR; m++) {
                pa[m] = 0.0f;
            }
            pa += TILE_MR;
        }
    }
    return packed_data;
}

#if USE_AVX2
#define KERNEL_STEP_PACKED_A(KK) \
    bL = _mm256_loadu_ps(pb + (KK) * TILE_NR); \
    bR = _mm256_loadu_ps(pb + (KK) * TILE_NR + 8); \
    va = _mm256_set1_ps(pa[(KK)*TILE_MR + 0]); c0L = _mm256_fmadd_ps(va, bL, c0L); c0R = _mm256_fmadd_ps(va, bR, c0R); \
    va = _mm256_set1_ps(pa[(KK)*TILE_MR + 1]); c1L = _mm256_fmadd_ps(va, bL, c1L); c1R = _mm256_fmadd_ps(va, bR, c1R); \
    va = _mm256_set1_ps(pa[(KK)*TILE_MR + 2]); c2L = _mm256_fmadd_ps(va, bL, c2L); c2R = _mm256_fmadd_ps(va, bR, c2R); \
    va = _mm256_set1_ps(pa[(KK)*TILE_MR + 3]); c3L = _mm256_fmadd_ps(va, bL, c3L); c3R = _mm256_fmadd_ps(va, bR, c3R); \
    va = _mm256_set1_ps(pa[(KK)*TILE_MR + 4]); c4L = _mm256_fmadd_ps(va, bL, c4L); c4R = _mm256_fmadd_ps(va, bR, c4R); \
    va = _mm256_set1_ps(pa[(KK)*TILE_MR + 5]); c5L = _mm256_fmadd_ps(va, bL, c5L); c5R = _mm256_fmadd_ps(va, bR, c5R);

/* prefetch 提示宏: 让下一段 pa/pb 提前进入 L1.
 * 64 字节 = 1 cache line. pa stride = TILE_MR*4 = 24B, 所以提前 8 步覆盖大约 192 B = 3 cache lines.
 * pb stride = TILE_NR*4 = 64B, 所以提前 4 步覆盖 256B = 4 cache lines. */
#define PF_AHEAD 8
#define PREFETCH_NEXT(KK) \
    _mm_prefetch((const char*)(pa + ((KK) + PF_AHEAD) * TILE_MR), _MM_HINT_T0); \
    _mm_prefetch((const char*)(pb + ((KK) + PF_AHEAD) * TILE_NR), _MM_HINT_T0); \
    _mm_prefetch((const char*)(pb + ((KK) + PF_AHEAD) * TILE_NR + 8), _MM_HINT_T0);

extern void micro_6x16_asm_packed_a(const float *pa, const float *pb, float *C, int ldc, int ck, int zero_mode);

static void micro_6x16_packed_a(const float * __restrict__ pa,
                                const float * __restrict__ pb,
                                float * __restrict__ C, int ldc,
                                int ck, int actual_m, int actual_n,
                                int zero_mode) {
    __m256 c0L, c0R, c1L, c1R, c2L, c2R, c3L, c3R, c4L, c4R, c5L, c5R;
    __m256 bL, bR, va;

    /* 对于边界分块，如果是非 zero_mode，先存到局部再累加，避免越界 load */
    int is_edge = (actual_m < TILE_MR || actual_n < TILE_NR);

    if (is_edge) {
        c0L=_mm256_setzero_ps(); c0R=_mm256_setzero_ps();
        c1L=_mm256_setzero_ps(); c1R=_mm256_setzero_ps();
        c2L=_mm256_setzero_ps(); c2R=_mm256_setzero_ps();
        c3L=_mm256_setzero_ps(); c3R=_mm256_setzero_ps();
        c4L=_mm256_setzero_ps(); c4R=_mm256_setzero_ps();
        c5L=_mm256_setzero_ps(); c5R=_mm256_setzero_ps();
    } else {
#ifdef USE_ASM
        micro_6x16_asm_packed_a(pa, pb, C, ldc, ck, zero_mode);
        return;
#else
        if (zero_mode) {
            c0L=_mm256_setzero_ps(); c0R=_mm256_setzero_ps();
            c1L=_mm256_setzero_ps(); c1R=_mm256_setzero_ps();
            c2L=_mm256_setzero_ps(); c2R=_mm256_setzero_ps();
            c3L=_mm256_setzero_ps(); c3R=_mm256_setzero_ps();
            c4L=_mm256_setzero_ps(); c4R=_mm256_setzero_ps();
            c5L=_mm256_setzero_ps(); c5R=_mm256_setzero_ps();
        } else {
            c0L=_mm256_loadu_ps(C);            c0R=_mm256_loadu_ps(C+8);
            c1L=_mm256_loadu_ps(C+ldc);        c1R=_mm256_loadu_ps(C+ldc+8);
            c2L=_mm256_loadu_ps(C+2*ldc);      c2R=_mm256_loadu_ps(C+2*ldc+8);
            c3L=_mm256_loadu_ps(C+3*ldc);      c3R=_mm256_loadu_ps(C+3*ldc+8);
            c4L=_mm256_loadu_ps(C+4*ldc);      c4R=_mm256_loadu_ps(C+4*ldc+8);
            c5L=_mm256_loadu_ps(C+5*ldc);      c5R=_mm256_loadu_ps(C+5*ldc+8);
        }
#endif
    }

    int k = 0;
    while (k + 3 < ck) {
        /* prefetch 提示: 提前 PF_AHEAD 步 (=8) 把 pa/pb 拉进 L1.
         * 在 4 步展开的开头各发一次 prefetch, 摊薄指令开销. */
        PREFETCH_NEXT(0);
        KERNEL_STEP_PACKED_A(0);
        KERNEL_STEP_PACKED_A(1);
        PREFETCH_NEXT(2);
        KERNEL_STEP_PACKED_A(2);
        KERNEL_STEP_PACKED_A(3);
        pa += 4 * TILE_MR;
        pb += 4 * TILE_NR;
        k += 4;
    }
    for (; k < ck; k++) {
        KERNEL_STEP_PACKED_A(0);
        pa += TILE_MR;
        pb += TILE_NR;
    }

    if (is_edge) {
        float out_block[TILE_MR * TILE_NR] __attribute__((aligned(32)));
        _mm256_storeu_ps(out_block, c0L);            _mm256_storeu_ps(out_block+8, c0R);
        _mm256_storeu_ps(out_block+1*16, c1L);       _mm256_storeu_ps(out_block+1*16+8, c1R);
        _mm256_storeu_ps(out_block+2*16, c2L);       _mm256_storeu_ps(out_block+2*16+8, c2R);
        _mm256_storeu_ps(out_block+3*16, c3L);       _mm256_storeu_ps(out_block+3*16+8, c3R);
        _mm256_storeu_ps(out_block+4*16, c4L);       _mm256_storeu_ps(out_block+4*16+8, c4R);
        _mm256_storeu_ps(out_block+5*16, c5L);       _mm256_storeu_ps(out_block+5*16+8, c5R);
        
        for (int m = 0; m < actual_m; m++) {
            for (int n = 0; n < actual_n; n++) {
                if (zero_mode) C[m * ldc + n] = out_block[m * 16 + n];
                else           C[m * ldc + n] += out_block[m * 16 + n];
            }
        }
    } else {
#ifndef USE_ASM
        _mm256_storeu_ps(C, c0L);            _mm256_storeu_ps(C+8, c0R);
        _mm256_storeu_ps(C+ldc, c1L);        _mm256_storeu_ps(C+ldc+8, c1R);
        _mm256_storeu_ps(C+2*ldc, c2L);      _mm256_storeu_ps(C+2*ldc+8, c2R);
        _mm256_storeu_ps(C+3*ldc, c3L);      _mm256_storeu_ps(C+3*ldc+8, c3R);
        _mm256_storeu_ps(C+4*ldc, c4L);      _mm256_storeu_ps(C+4*ldc+8, c4R);
        _mm256_storeu_ps(C+5*ldc, c5L);      _mm256_storeu_ps(C+5*ldc+8, c5R);
#endif
    }
}
#undef KERNEL_STEP_PACKED_A
#endif // USE_AVX2

static void __attribute__((unused)) micro_scalar_packed_a(const float * __restrict__ pa,
                                  const float * __restrict__ pb,
                                  float * __restrict__ C, int ldc,
                                  int ck, int cm, int cn, int zero_mode) {
    for (int m = 0; m < cm; m++) {
        for (int n = 0; n < cn; n++) {
            float sum = 0.0f;
            for (int k = 0; k < ck; k++) sum += pa[k*TILE_MR + m] * pb[k*TILE_NR+n];
            if (zero_mode) C[m*ldc+n] = sum;
            else           C[m*ldc+n] += sum;
        }
    }
}

void sgemm_nn_packed_a(int M, int N, int K,
                       const void *packed_A,
                       const float *B, int ldb,
                       float *C, int ldc) {
    if (M <= 0 || N <= 0 || K <= 0) return;

    const float *packed_a_flt = (const float*)packed_A;

    /* 小矩阵快速通道: N=1 (末尾特征空间尺寸为1的 1x1 卷积等) */
    if (N == 1) {
        for (int m0 = 0; m0 < M; m0 += TILE_MR) {
            int cm = (M - m0 < TILE_MR) ? (M - m0) : TILE_MR;
            const float *pa = packed_a_flt + (m0 / TILE_MR) * K * TILE_MR;
#if USE_AVX2
            __m256 sum = _mm256_setzero_ps();
            for (int k = 0; k < K; k++) {
                __m256 va = _mm256_loadu_ps(pa); pa += TILE_MR;
                __m256 vb = _mm256_set1_ps(B[k * ldb]);
                sum = _mm256_fmadd_ps(va, vb, sum);
            }
            float out[8] __attribute__((aligned(32)));
            _mm256_storeu_ps(out, sum);
            for (int m = 0; m < cm; m++) C[(m0 + m) * ldc] = out[m];
#else
            for (int m = 0; m < cm; m++) C[(m0 + m) * ldc] = 0.0f;
            for (int k = 0; k < K; k++) {
                float b_val = B[k * ldb];
                for (int m = 0; m < cm; m++) {
                    C[(m0 + m) * ldc] += pa[k * TILE_MR + m] * b_val;
                }
                pa += TILE_MR;
            }
#endif
        }
        return;
    }

    int use_par = 0;
#ifdef _OPENMP
    use_par = (M >= OMP_MIN_M);
#endif


    /* 分摊打包 B 缓冲，供并行任务安全只读共用.
     * 使用线程局部静态缓冲, 消除每次调用的 malloc/free 开销. */
    float *B_pack_shared = get_b_pack_buf((size_t)TILE_KC * TILE_NC);
    if (!B_pack_shared) return;

    for (int n0 = 0; n0 < N; n0 += TILE_NC) {
        int cn = (N-n0 < TILE_NC) ? (N-n0) : TILE_NC;
        int num_n_blocks = (cn + TILE_NR - 1) / TILE_NR;
        
        for (int k0 = 0; k0 < K; k0 += TILE_KC) {
            int ck = (K-k0 < TILE_KC) ? (K-k0) : TILE_KC;
            int zm = (k0 == 0) ? 1 : 0; 

            for (int nj = 0; nj < cn; nj += TILE_NR) {
                int cnr = (cn-nj < TILE_NR) ? (cn-nj) : TILE_NR;
                /* 单线程/串行预打包这一批次的 B 到共享段 */
                pack_b(B_pack_shared + nj * ck, B+k0*ldb+n0+nj, ldb, ck, cnr);
            }

            int num_m_blocks = (M + TILE_MR - 1) / TILE_MR;
            int total_tasks = num_m_blocks * num_n_blocks;

            #ifdef _OPENMP
            #pragma omp parallel for schedule(static) if(use_par)
            #endif
            for (int task = 0; task < total_tasks; task++) {
                int m0_idx = task / num_n_blocks;
                int nj_idx = task % num_n_blocks;
                
                int m0 = m0_idx * TILE_MR;
                int nj = nj_idx * TILE_NR;
                
                int cm = (M-m0 < TILE_MR) ? (M-m0) : TILE_MR;
                int cnr = (cn-nj < TILE_NR) ? (cn-nj) : TILE_NR;

                const float *pa = packed_a_flt + m0_idx * K * TILE_MR + k0 * TILE_MR;
                const float *pb = B_pack_shared + nj * ck;

                #if USE_AVX2
                micro_6x16_packed_a(pa, pb, C+m0*ldc+n0+nj, ldc, ck, cm, cnr, zm);
                #else
                micro_scalar_packed_a(pa, pb, C+m0*ldc+n0+nj, ldc, ck, cm, cnr, zm);
                #endif
            }
        }
    }
    /* B_pack_shared 是 thread-local static, 不需要释放 */
}
