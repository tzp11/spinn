#!/usr/bin/env python3
"""
深入分析 ONNX Runtime 的性能优势
对比 SPINN 找出具体差距
"""
import numpy as np
import onnxruntime as ort
import time
import sys

def analyze_model_execution(onnx_path, model_name):
    """分析模型执行细节"""
    print(f"\n{'='*70}")
    print(f"Analyzing: {model_name}")
    print(f"{'='*70}")
    
    # 创建 session 时启用性能分析
    sess_options = ort.SessionOptions()
    sess_options.enable_profiling = True
    sess_options.profile_file_prefix = f"ort_profile_{model_name}"
    
    # 获取执行提供者信息
    providers = ort.get_available_providers()
    print(f"\n[Available Providers]")
    for p in providers:
        print(f"  - {p}")
    
    # 创建 session
    print(f"\n[Creating Session]")
    t0 = time.time()
    sess = ort.InferenceSession(onnx_path, sess_options, providers=['CPUExecutionProvider'])
    load_time = time.time() - t0
    print(f"  Load time: {load_time*1000:.2f} ms")
    
    # 获取模型信息
    print(f"\n[Model Info]")
    print(f"  Inputs: {len(sess.get_inputs())}")
    for inp in sess.get_inputs():
        print(f"    - {inp.name}: {inp.shape} ({inp.type})")
    print(f"  Outputs: {len(sess.get_outputs())}")
    for out in sess.get_outputs():
        print(f"    - {out.name}: {out.shape} ({out.type})")
    
    # 准备输入
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    n = int(np.prod(shape))
    data = (np.arange(n, dtype=np.float32) / 1000.0).reshape(shape)
    
    print(f"\n[Input Data]")
    print(f"  Shape: {shape}")
    print(f"  Size: {n * 4 / 1024:.2f} KB")
    
    # Warmup
    print(f"\n[Warmup] (3 runs)")
    for i in range(3):
        sess.run(None, {inp.name: data})
    
    # Benchmark
    print(f"\n[Benchmark] (10 runs)")
    times = []
    for i in range(10):
        t0 = time.perf_counter()
        result = sess.run(None, {inp.name: data})
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f} ms")
    
    print(f"\n[Statistics]")
    print(f"  Best:   {min(times):.2f} ms")
    print(f"  Worst:  {max(times):.2f} ms")
    print(f"  Mean:   {np.mean(times):.2f} ms")
    print(f"  Median: {np.median(times):.2f} ms")
    print(f"  Std:    {np.std(times):.2f} ms")
    
    # 输出样例
    output = result[0].flatten()
    print(f"\n[Output Sample] (first 10)")
    print(f"  {' '.join([f'{v:.6f}' for v in output[:10]])}")
    
    # 性能分析文件
    print(f"\n[Profiling]")
    print(f"  Profile saved to: ort_profile_{model_name}_*.json")
    print(f"  Use Chrome tracing (chrome://tracing) to visualize")
    
    return min(times)

def compare_with_spinn(model_name, ort_time, spinn_time):
    """对比 ORT 和 SPINN 的性能"""
    print(f"\n{'='*70}")
    print(f"Performance Comparison: {model_name}")
    print(f"{'='*70}")
    
    ratio = spinn_time / ort_time
    print(f"\n  ORT:   {ort_time:.2f} ms")
    print(f"  SPINN: {spinn_time:.2f} ms")
    print(f"  Ratio: {ratio:.2f}x (SPINN is {ratio:.2f}x slower)")
    
    if ratio > 2:
        print(f"\n  ⚠️  SPINN is significantly slower!")
        print(f"  Possible reasons:")
        print(f"    1. Conv/GEMM implementation (ORT uses highly optimized kernels)")
        print(f"    2. Memory layout (ORT may use NCHWc for better cache locality)")
        print(f"    3. Operator fusion (ORT has more aggressive fusion)")
        print(f"    4. SIMD optimization (ORT uses AVX512/AVX2 more effectively)")
        print(f"    5. Multi-threading (ORT has better thread pool)")

def main():
    models = [
        ('SPINN/run_time/yolov10n.onnx', 'YOLOv10n', 595.90),
        ('SPINN/run_time/resnet101.onnx', 'ResNet101', 101.10),
    ]
    
    for onnx_path, name, spinn_time in models:
        try:
            ort_time = analyze_model_execution(onnx_path, name)
            compare_with_spinn(name, ort_time, spinn_time)
        except Exception as e:
            print(f"\nError analyzing {name}: {e}")
            continue

if __name__ == '__main__':
    main()
