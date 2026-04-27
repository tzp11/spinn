/*
 * gemm_kernel.h - 高性能 GEMM 内核
 * 
 * 参照 ORT MLAS sgemm.cpp 的架构：
 *   - 三级分块 Tiling (沿 N, K, M 方向)
 *   - B 矩阵 Pack 到连续缓存友好布局
 *   - 4 路循环展开
 *
 * C[M×N] = alpha * A[M×K] × B[K×N] + beta * C[M×N]
 */

#ifndef __GEMM_KERNEL_H__
#define __GEMM_KERNEL_H__

#include <stddef.h>

/*
 * 高性能 SGEMM：C = alpha * A * B + beta * C
 * 
 * A: M×K (行优先)
 * B: K×N (行优先)
 * C: M×N (行优先)
 * lda: A 的行跨度 (通常 = K)
 * ldb: B 的行跨度 (通常 = N)
 * ldc: C 的行跨度 (通常 = N)
 */
void sgemm_tiled(int M, int N, int K,
                 float alpha,
                 const float *A, int lda,
                 const float *B, int ldb,
                 float beta,
                 float *C, int ldc);

/*
 * 简化版：C = A * B (alpha=1, beta=0)
 */
void sgemm_nn(int M, int N, int K,
              const float *A, int lda,
              const float *B, int ldb,
              float *C, int ldc);

/* 
 * 静态权重预打包 (Offline Pack A)
 * 分配并返回打包后的 A 矩阵，大小可通过返回的 size 获取。
 */
void* sgemm_pack_a_offline(int M, int K, const float *A, int lda, size_t *out_size);

/*
 * 使用预打包的 A 的 SGEMM (alpha=1, beta=0)
 */
void sgemm_nn_packed_a(int M, int N, int K,
                       const void *packed_A,
                       const float *B, int ldb,
                       float *C, int ldc);

#endif /* __GEMM_KERNEL_H__ */
