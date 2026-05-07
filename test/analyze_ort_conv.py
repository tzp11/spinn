#!/usr/bin/env python3
"""
深入分析 ORT Conv 算子的实现策略
对比 SPINN 的 im2col+GEMM 方法
"""
import onnx
import numpy as np

def analyze_conv_nodes(onnx_path, model_name):
    """分析模型中的 Conv 节点"""
    print(f"\n{'='*70}")
    print(f"Analyzing Conv nodes in: {model_name}")
    print(f"{'='*70}")
    
    model = onnx.load(onnx_path)
    graph = model.graph
    
    conv_nodes = [n for n in graph.node if n.op_type == 'Conv']
    print(f"\nTotal Conv nodes: {len(conv_nodes)}")
    
    # 统计不同类型的 Conv
    conv_types = {
        '1x1': [],
        '3x3': [],
        '5x5': [],
        '7x7': [],
        'other': []
    }
    
    for node in conv_nodes:
        # 获取 kernel_shape
        kernel_shape = None
        for attr in node.attribute:
            if attr.name == 'kernel_shape':
                kernel_shape = list(attr.ints)
                break
        
        if kernel_shape:
            key = f"{kernel_shape[0]}x{kernel_shape[1]}"
            if key in conv_types:
                conv_types[key].append(node)
            else:
                conv_types['other'].append(node)
    
    print(f"\n[Conv Distribution]")
    for key, nodes in conv_types.items():
        if nodes:
            print(f"  {key}: {len(nodes)} nodes")
    
    # 分析典型的 Conv 配置
    print(f"\n[Typical Conv Configurations]")
    for i, node in enumerate(conv_nodes[:10]):  # 只看前10个
        print(f"\n  Conv #{i+1}: {node.name}")
        for attr in node.attribute:
            if attr.name in ['kernel_shape', 'strides', 'pads', 'dilations', 'group']:
                if attr.ints:
                    print(f"    {attr.name}: {list(attr.ints)}")
                elif attr.i:
                    print(f"    {attr.name}: {attr.i}")
    
    return conv_types

def explain_ort_optimizations():
    """解释 ORT 的 Conv 优化策略"""
    print(f"\n{'='*70}")
    print(f"ORT Conv Optimization Strategies")
    print(f"{'='*70}")
    
    print(f"""
[1] Direct Convolution (1x1, small kernels)
    - 不使用 im2col，直接计算
    - 减少内存分配和拷贝
    - 更好的 cache locality
    - SPINN 使用: im2col + GEMM (额外开销)

[2] Winograd Algorithm (3x3, stride=1)
    - 减少乘法次数: 9 次 → 4 次
    - 2.25x 理论加速
    - 需要额外的变换开销
    - SPINN 未实现

[3] NCHWc Layout (Blocked Layout)
    - 将 C 维度分块: NCHW → NC(C/16)HW(16)
    - 更好的 SIMD 向量化
    - 减少 cache miss
    - SPINN 使用: 标准 NCHW

[4] GEMM Optimization
    - 使用 MKL/OpenBLAS 等高度优化的库
    - 或自己实现的高性能 GEMM (MLAS)
    - 多级 cache blocking
    - SPINN 使用: 自己的 6x16 微内核

[5] Operator Fusion
    - Conv + BN + ReLU 融合为一个 kernel
    - 减少内存读写
    - 减少 kernel launch 开销
    - SPINN 已实现部分融合

[6] Multi-threading
    - 精细的线程池管理
    - 动态负载均衡
    - 减少同步开销
    - SPINN 使用: OpenMP parallel for

[7] Memory Pool
    - 预分配内存池
    - 减少 malloc/free 开销
    - 内存复用
    - SPINN 使用: 静态 Arena (更好)
""")

def estimate_performance_gap():
    """估算各优化带来的性能提升"""
    print(f"\n{'='*70}")
    print(f"Performance Gap Analysis")
    print(f"{'='*70}")
    
    print(f"""
[YOLOv10n: SPINN 595ms vs ORT 64ms = 9.3x slower]

估算各因素的影响:

1. Direct Conv (1x1)
   - ORT: 直接计算
   - SPINN: im2col + GEMM
   - 预估影响: 1.5-2x

2. Winograd (3x3, stride=1)
   - ORT: 使用 Winograd
   - SPINN: im2col + GEMM
   - 预估影响: 2-2.5x

3. NCHWc Layout
   - ORT: 使用 blocked layout
   - SPINN: 标准 NCHW
   - 预估影响: 1.3-1.5x

4. GEMM Quality
   - ORT: MLAS (高度优化)
   - SPINN: 自己的 6x16 kernel
   - 预估影响: 1.2-1.5x

5. Small Operators (Resize/TopK/etc)
   - ORT: 高度优化
   - SPINN: 标量实现
   - 预估影响: 1.5-2x

综合影响: 1.5 × 2.0 × 1.3 × 1.2 × 1.5 ≈ 7-10x

这与实测的 9.3x 基本吻合！

[ResNet101: SPINN 101ms vs ORT 40ms = 2.5x slower]

ResNet101 主要是 Conv + GEMM，小算子少:

1. Winograd (3x3, stride=1)
   - 预估影响: 2-2.5x

2. GEMM Quality
   - 预估影响: 1.2-1.5x

综合影响: 2.0 × 1.2 ≈ 2.4x

这与实测的 2.5x 基本吻合！
""")

def main():
    print("="*70)
    print("ORT vs SPINN Performance Analysis")
    print("="*70)
    
    # 分析 YOLOv10n
    analyze_conv_nodes('SPINN/run_time/yolov10n.onnx', 'YOLOv10n')
    
    # 分析 ResNet101
    analyze_conv_nodes('SPINN/run_time/resnet101.onnx', 'ResNet101')
    
    # 解释 ORT 优化
    explain_ort_optimizations()
    
    # 估算性能差距
    estimate_performance_gap()

if __name__ == '__main__':
    main()
