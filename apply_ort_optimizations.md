# ORT Conv 实现关键点（从源码分析）

## 1. MLAS Conv 实现路径选择

ORT 的 Conv 实现在 `onnxruntime/core/mlas/lib/convolve.cpp`：

```cpp
// 路径选择逻辑
if (kernel_h == 1 && kernel_w == 1) {
    // 1×1 Conv: 直接 GEMM，无 im2col
    MlasGemm(...);
} else if (kernel_h == 3 && kernel_w == 3 && stride == 1) {
    // 3×3 stride=1: Winograd F(2×2, 3×3) 或 F(4×4, 3×3)
    MlasConvWinograd(...);
} else {
    // 通用路径: im2col + GEMM
    MlasIm2Col(...);
    MlasGemm(...);
}
```

## 2. MLAS GEMM 关键优化

在 `onnxruntime/core/mlas/lib/sgemm.cpp`：

```cpp
// 关键参数
#define MLAS_SGEMM_STRIDEN  128  // N 维度 blocking
#define MLAS_SGEMM_STRIDEK  256  // K 维度 blocking

// 微内核大小（AVX2）
#define MLAS_SGEMM_KERNEL_M  6
#define MLAS_SGEMM_KERNEL_N  16

// 三级 blocking
for (size_t n = 0; n < N; n += STRIDEN) {
    for (size_t k = 0; k < K; k += STRIDEK) {
        // Pack B panel
        MlasSgemmCopyPackB(...);
        
        for (size_t m = 0; m < M; m += STRIDEM) {
            // 微内核
            MlasSgemmKernel6x16(...);
        }
    }
}
```

## 3. 关键差异

| 特性 | ORT MLAS | SPINN 当前 |
|:---|:---|:---|
| N blocking | 128 | 无 |
| K blocking | 256 | 无（全量） |
| M blocking | 动态 | 无 |
| B packing | 每个 K panel | 一次性 |
| Prefetch | 多级 | 简单 |
| 线程并行 | 按 N 分块 | 按 M 分块 |

## 4. 需要应用的优化

1. **设置 OMP_NUM_THREADS=4**（已完成）
2. **改进 GEMM blocking**
3. **优化 B packing 策略**
4. **启用 Winograd**（需验证正确性）
