#!/usr/bin/env python3
"""
ORT Benchmark - ResNet-101 推理性能对比
用同一个 ONNX 模型，分别测试 1 线程和 4 线程的 ORT 推理时间。
同时输出前 10 个结果值，方便与 SPINN 对比正确性。
"""
import onnxruntime as ort
import numpy as np
import time
import sys

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "SPINN/run_time/resnet101.onnx"
N_WARMUP = 3
N_RUNS = 10

def benchmark(model_path, num_threads):
    """运行 ORT benchmark"""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(model_path, opts, providers=['CPUExecutionProvider'])
    
    # 获取输入信息
    input_info = session.get_inputs()[0]
    input_shape = [d if isinstance(d, int) else 1 for d in input_info.shape]
    input_data = np.random.randn(*input_shape).astype(np.float32)
    
    input_name = input_info.name
    
    # Warmup
    for _ in range(N_WARMUP):
        session.run(None, {input_name: input_data})
    
    # Test
    times = []
    result = None
    for i in range(N_RUNS):
        start = time.perf_counter()
        outputs = session.run(None, {input_name: input_data})
        end = time.perf_counter()
        times.append(end - start)
        if result is None:
            result = outputs[0]
    
    times.sort()
    # 取中位数 (去掉最快和最慢)
    median_time = times[N_RUNS // 2]
    min_time = times[0]
    avg_time = sum(times) / len(times)
    
    return min_time, median_time, avg_time, result

print(f"ONNX Runtime v{ort.__version__}")
print(f"Model: {MODEL_PATH}")
print(f"Warmup: {N_WARMUP}, Runs: {N_RUNS}")
print()

for threads in [1, 2, 4, 8]:
    min_t, med_t, avg_t, result = benchmark(MODEL_PATH, threads)
    flat = result.flatten()
    first10 = ' '.join(f'{v:.6f}' for v in flat[:10])
    print(f"[{threads}T]  min={min_t*1000:.1f}ms  median={med_t*1000:.1f}ms  avg={avg_t*1000:.1f}ms")
    if threads == 1:
        print(f"      Output: {first10}")
    print()
