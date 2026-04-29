# SPINN 性能优化进展报告

> 最后更新: 2026-04-28  
> 测试环境: Linux x86_64, AVX2+FMA, 4 核 (OMP_NUM_THREADS=4)

---

## 1. 项目概述

**SPINN** (Space Payload Inference Neural Network) — 面向星载嵌入式的纯 C 神经网络推理引擎。  
核心特性: 静态内存规划 (零 malloc)、ENNF 二进制格式、图优化融合、AVX2 SIMD 算子。

**对标**: ONNX Runtime (ORT) 1.17+, 同硬件同线程数公平对比。

---

## 2. 整体性能对比

### 2.1 推理延迟 (Best / Avg, ms)

| 模型 | 输入尺寸 | SPINN Best | SPINN Avg | ORT Best | ORT Avg | SPINN/ORT (Best) |
|------|----------|-----------|----------|---------|---------|-------------------|
| ResNet101 | 1×3×224×224 | **87.7** | 92.3 | **68.6** | 81.5 | 1.28× |
| YOLOv10n | 1×3×640×640 | **161.7** | 166.3 | **46.0** | 71.7 | 3.52× |

> ResNet101 以 Conv+GEMM 为主, SPINN 已接近 ORT (1.28×).  
> YOLOv10n 差距较大, 主要因小算子 (Resize/TopK/Split/Sigmoid 等) 仍为标量实现.

### 2.2 算子级 Profile (YOLOv10n, 单次推理 5 次取 Best)

| 算子 | 调用次数 | 总耗时 ms | 单次 ms | 占比 |
|------|---------|----------|---------|------|
| Conv | 664 | 1138.7 | 1.72 | 82.9% |
| Softmax | 16 | 35.8 | 2.24 | 2.6% |
| Concat | 168 | 31.2 | 0.19 | 2.3% |
| Add | 64 | 30.3 | 0.47 | 2.2% |
| MaxPool | 24 | 24.4 | 1.02 | 1.8% |
| TopK | 16 | 20.4 | 1.28 | 1.5% |
| Resize | 16 | 20.4 | 1.28 | 1.5% |
| Sigmoid | 8 | 15.2 | 1.90 | 1.1% |
| Transpose | 32 | 13.9 | 0.43 | 1.0% |
| Split | 104 | 13.1 | 0.13 | 1.0% |
| Mul | 16 | 9.2 | 0.57 | 0.7% |
| MatMul | 16 | 6.5 | 0.41 | 0.5% |

### 2.3 算子级 Profile (ResNet101)

| 算子 | 调用次数 | 总耗时 ms | 单次 ms | 占比 |
|------|---------|----------|---------|------|
| Conv | 832 | 915.9 | 1.10 | 93.3% |
| Gemm | 8 | 50.8 | 6.35 | 5.2% |
| MaxPool | 8 | 15.0 | 1.88 | 1.5% |
| ReduceMean | 8 | 0.5 | 0.06 | 0.0% |

---

## 3. 已完成的优化及效果

### 3.1 Transpose: 嵌套循环 + 尾部 memcpy (8× 加速)

**文件**: `run_time/ops/shape/transpose.c`  
**提交**: `69006ed`

| 指标 | 优化前 | 优化后 | 改进 |
|------|--------|--------|------|
| YOLOv10n Transpose 总耗时 | ~120 ms (8.0%) | 14 ms (1.0%) | **-88%** |
| 单次调用 | 3.75 ms | 0.43 ms | **-88%** |

**技术要点**:
- 原实现: 每个元素做 ndim 次 `div/mod` 计算坐标, O(N×D) 除法
- 新实现: 嵌套计数器 + 进位递增, 完全消除 div/mod
- 检测尾部恒等维度 (perm[d]==d 的最长后缀), 内层 memcpy
- 全局恒等 perm 走 memcpy 快路径
- 输出顺序遍历, 减少写入侧 cache miss

### 3.2 Softmax: AVX2 SIMD + Cephes expf (25% 加速)

**文件**: `run_time/ops/activation/softmax.c`, `run_time/ops/activation/simd_math.h`  
**提交**: `4a5de5b`

| 指标 | 优化前 | 优化后 | 改进 |
|------|--------|--------|------|
| YOLOv10n Softmax 总耗时 | 47 ms (3.3%) | 36 ms (2.6%) | **-25%** |
| 单次调用 | 2.94 ms | 2.24 ms | **-24%** |

**技术要点**:
- `inner=1` (axis=最后维, 内存连续) fast path: AVX2 三遍扫 (max→exp+sum→scale)
- `expf` 用 Cephes 标准实现: ln(2) 高低位分解 + 7 阶多项式, ~1 ULP 精度
- 横向归约用 `_mm256_extractf128` + `_mm_max/hadd`
- 精度验证: max_rel=3.5e-7 (与 libm expf 等价)

### 3.3 Sigmoid: AVX2 SIMD (24% 加速)

**文件**: `run_time/ops/activation/sigmoid.c`, `run_time/ops/activation/simd_math.h`  
**提交**: `4a5de5b` (同 commit)

| 指标 | 优化前 | 优化后 | 改进 |
|------|--------|--------|------|
| YOLOv10n Sigmoid 总耗时 | ~20 ms (1.5%) | 15 ms (1.1%) | **-24%** |
| 单次调用 | 2.51 ms | 1.90 ms | **-24%** |

**技术要点**:
- 复用 `simd_math.h` 中 `spinn_sigmoid256_ps()` (8 路 AVX2)
- `sigmoid(x) = 1 / (1 + exp(-x))`, exp 部分复用 `spinn_exp256_ps`
- 标量 fallback 处理尾部元素

### 3.4 GEMM: Thread-local B-pack + Prefetch

**文件**: `run_time/ops/mm/gemm_kernel.c`  
**提交**: `6ded981`

| 指标 | 改进说明 |
|------|---------|
| B-pack 缓冲区 | `thread_local` 静态缓冲, 消除每次 `malloc/free` |
| Prefetch | 微内核循环插入 `_mm_prefetch` 预取下一块 B 数据 |
| ResNet101 GEMM | 8 次 Gemm 调用, 单次 6.35 ms, 合理 |

### 3.5 图优化: Conv+Add(+ReLU) + Conv+SiLU 融合

**文件**: `graph_opt.c`, `run_time/ops/conv/conv2d.c`  
**提交**: `5cb95b2`

| 融合类型 | 说明 |
|---------|------|
| Conv+Add(+ReLU) | 残差分支融合, 消除 Add 算子开销, 含拓扑序检查 |
| Conv+SiLU | YOLOv10n 激活融合, conv2d 内直接计算 `x * sigmoid(x)` |

**关键修复**: 残差输入来自后续节点 → 添加拓扑序检查, 防止错误融合

### 3.6 其他基础设施

| 功能 | 文件 | 提交 |
|------|------|------|
| Per-op Profile | `spinn_runtime.c`, `main.c` | `33005aa` |
| OMP 线程数修正 | `bench_compare.py` | `2087fd1` |
| RSS 同口径对比 | `rss_compare.py` | `0489d39` |
| ENNF→C 转译器 | `ennf2c/ennf2c.c` | `2302caa`, `32101ac` |
| M_max 内存约束 | `spinn_runtime.c/h` | `0e864d8` |

---

## 4. 待完成优化 (按优先级排序)

### P0: Conv 算子进一步优化 (占 82.9%, 最大瓶颈)

**当前**: im2col + GEMM, 单次 1.72 ms (YOLOv10n) / 1.10 ms (ResNet101)

| 优化项 | 预期收益 | 难度 | 状态 |
|--------|---------|------|------|
| MC 分块 (L2 cache blocking) | 10-20% | 中 | 待做 |
| A 矩阵 offline packing | 5-15% | 中 | 待做 |
| 小 Conv 跳过 im2col (1×1, 3×3 direct) | 20-40% (小层) | 高 | 待做 |
| Winograd (3×3, stride=1) | 2-3× (3×3 层) | 高 | 待做 |

### P1: GEMM 进一步优化

| 优化项 | 预期收益 | 难度 | 状态 |
|--------|---------|------|------|
| 增大 KC/NC blocking (匹配 L3) | 5-10% | 低 | 待做 |
| A 矩阵 packing (运行时) | 5-10% | 中 | 待做 |

### P2: 小算子 SIMD 化

| 算子 | 当前耗时 | 占比 | 优化方案 | 状态 |
|------|---------|------|---------|------|
| Resize | 20 ms | 1.5% | 双线性插值 AVX2 | 待做 |
| TopK | 20 ms | 1.5% | 部分排序优化 | 待做 |
| MaxPool | 24 ms | 1.8% | AVX2 向量化 | 待做 |
| Add | 30 ms | 2.2% | AVX2 向量化 | 待做 |
| Mul | 9 ms | 0.7% | AVX2 向量化 | 待做 |
| Split | 13 ms | 1.0% | memcpy 优化 | 待做 |

### P3: 小 Conv OpenMP 调度优化

**问题**: 小 Conv 层 (如 1×1, 8×8) 并行开销 > 计算量, OpenMP 反而拖慢  
**方案**: 设阈值, 小层串行; 大层按 OC 划分, 减少 barrier  
**预期**: 减少 5-10% 总耗时  
**状态**: 待做

### P4: NCHW ↔ NCHWc 布局优化

**问题**: YOLOv10n 有 32 次 Transpose (NCHW→NC4HW4 等), 即使单次 0.43ms 仍占 1%  
**方案**: 全图 NCHWc 布局传播, Conv 内部用 NCHWc, 消除 Transpose  
**预期**: 消除大部分 Transpose 调用  
**状态**: 待做

---

## 5. 优化效果汇总 (YOLOv10n)

| 阶段 | Best (ms) | 相对 ORT | 主要改进 |
|------|-----------|---------|---------|
| 初始 (无优化) | ~250 | 5.4× | 基线 |
| + GEMM B-pack + Prefetch | ~210 | 4.6× | GEMM 优化 |
| + 图优化融合 | ~200 | 4.3× | Conv+Add/SiLU 融合 |
| + Transpose 优化 | ~170 | 3.7× | 嵌套循环+memcpy |
| + Softmax/Sigmoid SIMD | ~162 | 3.5× | AVX2 expf |
| **ORT 基线** | **46** | **1.0×** | — |

> **差距分析**: SPINN 3.5× 慢于 ORT, 主因:
> 1. Conv 82.9% — ORT 用 Winograd + direct conv + NCHWc, SPINN 仍用 im2col+GEMM
> 2. 小算子未 SIMD 化 — Resize/TopK/MaxPool/Add 等占 ~10%
> 3. OpenMP 调度 — ORT 有精细的线程池, SPINN 用粗粒度 parallel for

---

## 6. 关键文件索引

| 文件 | 功能 |
|------|------|
| `run_time/spinn_runtime.c/h` | 核心推理引擎, 内存规划, per-op profile |
| `run_time/ops/mm/gemm_kernel.c` | GEMM 6×16 AVX2 微内核, B-pack, prefetch |
| `run_time/ops/conv/conv2d.c` | Conv2D + fused post-ops (ReLU/SiLU/Add) |
| `run_time/ops/activation/softmax.c` | Softmax AVX2 fast path |
| `run_time/ops/activation/sigmoid.c` | Sigmoid AVX2 |
| `run_time/ops/activation/simd_math.h` | 共享 SIMD expf/sigmoid 实现 |
| `run_time/ops/shape/transpose.c` | Transpose 嵌套循环优化 |
| `graph_opt.c/h` | 图优化: Conv+BN/ReLU/Add/SiLU 融合 |
| `ennf2c/ennf2c.c` | ENNF→C 静态代码转译器 |
| `ennf2c/verify_three_way.py` | 三方数值验证 (ORT/SPINN/Generated) |
| `ennf2c/bench_compare.py` | 性能+RSS 对比脚本 |

---

## 7. 下一步行动 (优先级排序)

1. **Conv Winograd / direct conv** — 最大收益点, 3×3 stride=1 层可用 Winograd 2-3× 加速
2. **小 Conv OpenMP 调度** — 设阈值跳过小层并行, 减少开销
3. **Add/Mul/MaxPool AVX2** — 简单向量化, 每个约 10-20 行改动
4. **NCHWc 布局传播** — 消除 Transpose, 需改 Conv/BN/ReLU 等算子
5. **Resize/TopK 优化** — 需要算法级改进
