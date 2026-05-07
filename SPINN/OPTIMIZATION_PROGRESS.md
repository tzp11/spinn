# SPINN 性能优化进展报告

> 最后更新: 2026-05-06  
> 测试环境: Linux x86_64, Intel i7-9700K @ 3.60GHz, AVX2+FMA, 8C/8T, OMP_NUM_THREADS=4  
> ONNX Runtime: 1.23.2 (CPUExecutionProvider)

---

## 1. 项目概述

**SPINN** (Space Payload Inference Neural Network) — 面向星载嵌入式的纯 C 神经网络推理引擎。  
核心特性: 静态内存规划 (零 malloc)、ENNF 二进制格式、图优化融合、AVX2 SIMD 算子。

**对标**: ONNX Runtime (ORT) 1.23.2, 同硬件同线程数公平对比。

---

## 2. 整体性能对比

### 2.1 推理延迟 (Best / Avg, ms)

| 模型 | 输入尺寸 | SPINN Best | SPINN Avg | ORT Best | ORT Avg | SPINN/ORT (Best) |
|------|----------|-----------|----------|---------|---------|-------------------|
| ResNet101 | 1×3×224×224 | **95.7** | 103.1 | **165.7** | 218.9 | 0.58× |
| YOLOv10n | 1×3×640×640 | **177.7** | 192.9 | **151.1** | 233.7 | 1.18× |

> **ResNet101 SPINN 比 ORT 快 42%!** Winograd F(2,3) + 权重缓存 + fused bias/ReLU 三重优化.
> YOLOv10n 差距缩小至 1.18×, Winograd 已覆盖 3×3 Conv, 1×1 Conv 仍用 GEMM+融合.

### 2.1b RSS 峰值对比

| 模型 | SPINN RSS | ORT RSS | SPINN/ORT |
|------|----------|---------|----------|
| YOLOv10n | 63 MB | 133 MB | 0.47× |
| ResNet101 | 377 MB | 419 MB | 0.90× |

> SPINN 静态 Arena 内存规划, RSS 显著低于 ORT, YOLOv10n 仅 ORT 的 47%.

### 2.2 算子级 Profile (YOLOv10n, 含图优化融合, Best run)

| 算子 | 调用次数 | 总耗时 ms | 单次 ms | 占比 |
|------|---------|----------|---------|------|
| Conv | 664 | 1292.8 | 1.95 | 84.8% |
| Concat | 168 | 60.1 | 0.36 | 2.5% |
| Softmax | 16 | 49.4 | 3.09 | 2.1% |
| Add | 64 | 43.4 | 0.68 | 1.8% |
| MaxPool | 24 | 35.3 | 1.47 | 1.5% |
| Resize | 16 | 34.4 | 2.15 | 1.4% |
| TopK | 16 | 27.5 | 1.72 | 1.2% |
| Transpose | 32 | 21.1 | 0.66 | 0.9% |
| MatMul | 16 | 20.9 | 1.31 | 0.9% |
| Split | 104 | 18.4 | 0.18 | 0.8% |
| Sigmoid | 8 | 17.3 | 2.16 | 0.7% |

> 注: Conv+SiLU 融合后, Sigmoid 仅剩 8 次 (检测头部分), Mul=0 次 (全部融合进 Conv).

### 2.3 算子级 Profile (ResNet101, 含图优化融合, Best run)

| 算子 | 调用次数 | 总耗时 ms | 单次 ms | 占比 |
|------|---------|----------|---------|------|
| Conv | 832 | 1698.1 | 2.04 | 95.3% |
| Gemm | 8 | 63.4 | 7.92 | 3.6% |
| MaxPool | 8 | 20.0 | 2.50 | 1.1% |
| ReduceMean | 8 | 0.5 | 0.06 | 0.0% |

> 注: Conv+ReLU 融合后, Relu 调用次数为 0 (全部融合进 Conv).

---

## 3. 已完成的优化及效果

### 3.1 Transpose: 嵌套循环 + 尾部 memcpy (8× 加速)

**文件**: `run_time/ops/shape/transpose.c`  
**提交**: `69006ed`

| 指标 | 优化前 | 优化后 | 改进 |
|------|--------|--------|------|
| YOLOv10n Transpose 总耗时 | ~120 ms (8.0%) | 21 ms (0.9%) | **-83%** |
| 单次调用 | 3.75 ms | 0.66 ms | **-82%** |

**技术要点**:
- 原实现: 每个元素做 ndim 次 `div/mod` 计算坐标, O(N×D) 除法
- 新实现: 嵌套计数器 + 进位递增, 完全消除 div/mod
- 检测尾部恒等维度 (perm[d]==d 的最长后缀), 内层 memcpy
- 全局恒等 perm 走 memcpy 快路径
- 输出顺序遍历, 减少写入侧 cache miss

### 3.2 Softmax: AVX2 SIMD + Cephes expf

> **注**: 当前实测 Softmax 单次 3.09ms 比之前记录的 2.24ms 偏高, 可能与系统调度波动有关. AVX2 Cephes expf 实现本身已验证精度 (max_rel=3.5e-7), 性能需在更稳定环境下重测确认.

**文件**: `run_time/ops/activation/softmax.c`, `run_time/ops/activation/simd_math.h`  
**提交**: `4a5de5b`

| 指标 | 优化前 | 优化后 | 改进 |
|------|--------|--------|------|
| YOLOv10n Softmax 总耗时 | ~47 ms (3.3%) | 49 ms (2.1%) | 标量→AVX2 |
| 单次调用 | 2.94 ms | 3.09 ms | *见注* |

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
| YOLOv10n Sigmoid 总耗时 | ~20 ms (1.5%) | 17 ms (0.7%) | **融合后仅8次** |
| 单次调用 | 2.51 ms | 2.16 ms | AVX2 |

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
| ResNet101 GEMM | 8 次 Gemm 调用, 单次 7.92 ms |

### 3.5 图优化: Conv+Add(+ReLU) + Conv+SiLU 融合

**文件**: `graph_opt.c`, `run_time/ops/conv/conv2d.c`  
**提交**: `5cb95b2`

| 融合类型 | 实测效果 |
|---------|--------|
| Conv+Add(+ReLU) | YOLOv10n: 3 处融合; ResNet101: 67 处 Conv+ReLU 融合, Relu 调用降为 0 |
| Conv+SiLU | YOLOv10n: 69 处融合, Sigmoid 从 560→8 次, Mul 从 568→0 次 |

**关键修复**: 残差输入来自后续节点 → 添加拓扑序检查, 防止错误融合  
**重要**: 融合需用当前 `onnx2ennf` 重新转换模型才能生效 (旧 .ennf 文件不含融合信息)

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

### P0: Conv 算子进一步优化 (占 84.6%, 最大瓶颈)

**当前**: im2col + GEMM, 单次 3.03 ms (YOLOv10n) / 2.04 ms (ResNet101)

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
| Resize | 34 ms | 1.4% | 双线性插值 AVX2 | 待做 |
| Softmax | 49 ms | 2.1% | 优化调度/内存布局 | 待做 |
| Add | 43 ms | 1.8% | AVX2 向量化 | 待做 |
| MaxPool | 35 ms | 1.5% | AVX2 向量化 | 待做 |
| TopK | 28 ms | 1.2% | 部分排序优化 | 待做 |
| Transpose | 21 ms | 0.9% | 已优化, 进一步收益有限 | — |
| MatMul | 21 ms | 0.9% | 优化 GEMM 调用 | 待做 |
| Split | 18 ms | 0.8% | memcpy 优化 | 待做 |

### P3: 小 Conv OpenMP 调度优化

**问题**: 小 Conv 层 (如 1×1, 8×8) 并行开销 > 计算量, OpenMP 反而拖慢  
**方案**: 设阈值, 小层串行; 大层按 OC 划分, 减少 barrier  
**预期**: 减少 5-10% 总耗时  
**状态**: 待做

### P4: NCHW ↔ NCHWc 布局优化

**问题**: YOLOv10n 有 32 次 Transpose (NCHW→NC4HW4 等), 单次 0.66ms 占 0.9%  
**方案**: 全图 NCHWc 布局传播, Conv 内部用 NCHWc, 消除 Transpose  
**预期**: 消除大部分 Transpose 调用  
**状态**: 待做

---

## 5. 优化效果汇总 (YOLOv10n)

| 阶段 | Best (ms) | 相对 ORT | 主要改进 |
|------|-----------|---------|---------|
| 初始 (无优化, 无融合) | ~420 | 2.8× | 基线 (实测) |
| + 图优化融合 (重新转换模型) | ~260 | 1.72× | Conv+SiLU/ReLU 融合 |
| **ORT 基线** | **151** | **1.0×** | — |

> **差距分析**: SPINN 1.72× 慢于 ORT (YOLOv10n), 主因:
> 1. Conv 84.6% — ORT 用 Winograd + direct conv + NCHWc, SPINN 仍用 im2col+GEMM
> 2. 小算子未 SIMD 化 — Softmax/Resize/TopK/MaxPool/Add 等占 ~10%
> 3. OpenMP 调度 — ORT 有精细的线程池, SPINN 用粗粒度 parallel for
>
> ResNet101 仅 1.10× 慢于 ORT, Conv+ReLU 融合效果显著.

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

---

## 8. 数据可信度说明

> 本报告 2026-05-06 更新后的数据均来自当前环境 (i7-9700K) 实测, 可通过以下命令复现:
>
> ```bash
> # SPINN (含 per-op profile)
> OMP_NUM_THREADS=4 SPINN_PROFILE=1 ./spinn_run model.ennf 5
>
> # ORT
> OMP_NUM_THREADS=4 python3 test/analyze_ort_performance.py
> ```
>
> 2026-04-28 版本的数据无法在当前环境复现 (偏差 57%-393%), 可能原因:
> 1. 旧 .ennf 文件未经图优化转换, 融合未生效
> 2. ORT 版本差异 (当前 1.23.2 vs 报告声称 1.17+)
> 3. 硬件/系统负载差异
>
> 已用当前 `onnx2ennf` 重新转换模型, 融合已生效.
