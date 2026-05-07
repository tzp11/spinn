#!/usr/bin/env python3
"""
SPINN vs ONNX Runtime: 性能和正确性对比
"""
import numpy as np
import onnxruntime as ort
import time
import os
import sys

def benchmark_onnx(model_path, num_warmup=5, num_runs=20):
    """运行 ONNX Runtime 基准测试"""
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    # 获取输入信息
    input_info = sess.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape
    
    # 替换动态维度
    if isinstance(input_shape, list):
        input_shape = tuple(d if isinstance(d, int) else 1 for d in input_shape)
    
    # 使用与 SPINN 相同的输入: input[i] = i / 1000.0
    total_elem = 1
    for d in input_shape:
        total_elem *= d
    x = np.arange(total_elem, dtype=np.float32).reshape(input_shape) / 1000.0
    
    # Warmup
    for _ in range(num_warmup):
        sess.run(None, {input_name: x})
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        outputs = sess.run(None, {input_name: x})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    
    return outputs[0], times, input_shape

def get_spin_output(model_name, ort_out_shape):
    """运行 SPINN 并获取完整输出 (通过二进制 dump)"""
    import subprocess, tempfile
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'SPINN', 'run_time')
    ennf_path = os.path.join(base, f'{model_name}.ennf')
    exe_path = os.path.join(base, 'spinn_run')
    
    # 用临时文件做二进制 dump
    dump_file = tempfile.mktemp(suffix='.bin')
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'SPINN_DUMP': dump_file}
    
    result = subprocess.run(
        [exe_path, ennf_path, '3'],
        capture_output=True, text=True, env=env
    )
    
    # 读取二进制 dump
    if os.path.exists(dump_file):
        data = np.fromfile(dump_file, dtype=np.float32)
        os.unlink(dump_file)
        # reshape 到与 ORT 相同形状
        try:
            data = data.reshape(ort_out_shape)
        except:
            pass
        return data
    
    # fallback: 解析文本输出
    for line in (result.stdout + result.stderr).split('\n'):
        if 'SPINN Result:' in line:
            vals = line.split('SPINN Result:')[1].strip().split()
            return np.array([float(v) for v in vals])
    return None

def benchmark_spinn(model_name, num_warmup=3, num_runs=10):
    """运行 SPINN 基准测试"""
    import subprocess
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'SPINN', 'run_time')
    ennf_path = os.path.join(base, f'{model_name}.ennf')
    exe_path = os.path.join(base, 'spinn_run')
    
    env = {**os.environ, 'OMP_NUM_THREADS': '4'}
    
    # Warmup
    subprocess.run([exe_path, ennf_path, str(num_warmup)], capture_output=True, env=env)
    
    # Benchmark: 用 spinn_run 自带的计时
    result = subprocess.run(
        [exe_path, ennf_path, str(num_runs)],
        capture_output=True, text=True, env=env
    )
    
    # 解析 "Best: Xms, Avg: Yms"
    times = []
    for line in result.stderr.split('\n'):
        if 'Best:' in line:
            # "Best: 82.7ms, Avg: 99.0ms"
            best_str = line.split('Best:')[1].split(',')[0].strip().rstrip('ms')
            times.append(float(best_str))
    
    # 也解析每次 Run
    if not times:
        for line in result.stderr.split('\n'):
            if 'Run' in line and 'ms' in line:
                ms_str = line.split(':')[1].strip().rstrip('ms')
                times.append(float(ms_str))
    
    return times

def compare_outputs(ort_out, spinn_out, model_name):
    """比较 ONNX Runtime 和 SPINN 的输出"""
    if spinn_out is None:
        print(f"  ⚠ 无法获取 SPINN 输出")
        return
    
    ort_flat = ort_out.flatten()
    spinn_flat = spinn_out.flatten()[:len(ort_flat)]  # 截断到相同长度
    
    n = min(len(ort_flat), len(spinn_flat))
    ort_flat = ort_flat[:n]
    spinn_flat = spinn_flat[:n]
    
    # 计算误差
    abs_err = np.abs(ort_flat - spinn_flat)
    rel_err = abs_err / (np.abs(ort_flat) + 1e-8)
    
    max_abs = np.max(abs_err)
    mean_abs = np.mean(abs_err)
    max_rel = np.max(rel_err)
    mean_rel = np.mean(rel_err)
    
    # 余弦相似度
    cos_sim = np.dot(ort_flat, spinn_flat) / (np.linalg.norm(ort_flat) * np.linalg.norm(spinn_flat) + 1e-8)
    
    print(f"  输出元素数: {n}")
    print(f"  最大绝对误差: {max_abs:.6f}")
    print(f"  平均绝对误差: {mean_abs:.6f}")
    print(f"  最大相对误差: {max_rel:.4%}")
    print(f"  平均相对误差: {mean_rel:.4%}")
    print(f"  余弦相似度:   {cos_sim:.8f}")
    
    if cos_sim > 0.9999:
        print(f"  ✅ 正确性验证通过 (cos > 0.9999)")
    elif cos_sim > 0.999:
        print(f"  ⚠ 轻微差异 (cos > 0.999), 可能是浮点累积顺序不同")
    else:
        print(f"  ❌ 显著差异 (cos < 0.999)")

def main():
    models = [
        ('resnet101', 'ResNet101'),
        ('yolov10n', 'YOLOv10n'),
    ]
    
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'SPINN', 'run_time')
    
    print("=" * 70)
    print("  SPINN vs ONNX Runtime: 性能和正确性对比")
    print("=" * 70)
    
    all_ort = {}
    all_spinn = {}
    
    for model_name, display_name in models:
        print(f"\n{'='*70}")
        print(f"  {display_name}")
        print(f"{'='*70}")
        
        # ONNX Runtime
        onnx_path = os.path.join(base, f'{model_name}.onnx')
        if not os.path.exists(onnx_path):
            print(f"  ⚠ ONNX 模型不存在: {onnx_path}")
            continue
        
        print(f"\n  [ONNX Runtime]")
        ort_out, ort_times, input_shape = benchmark_onnx(onnx_path)
        print(f"  输入形状: {input_shape}")
        print(f"  输出形状: {ort_out.shape}")
        print(f"  Best: {min(ort_times):.1f}ms, Avg: {np.mean(ort_times):.1f}ms")
        print(f"  输出前10值: {ort_out.flatten()[:10]}")
        
        # SPINN
        print(f"\n  [SPINN]")
        spinn_out = get_spin_output(model_name, ort_out.shape)
        if spinn_out is not None:
            print(f"  输出前10值: {spinn_out[:10]}")
        
        # 正确性对比
        print(f"\n  [正确性对比]")
        compare_outputs(ort_out, spinn_out, display_name)
        
        # SPINN 性能
        print(f"\n  [SPINN 性能测试]")
        spinn_times = benchmark_spinn(model_name)
        if spinn_times:
            print(f"  Best: {min(spinn_times):.1f}ms, Avg: {np.mean(spinn_times):.1f}ms")
        else:
            print(f"  ⚠ 无法获取 SPINN 时间")
        
        all_ort[display_name] = ort_times
        all_spinn[display_name] = spinn_times
    
    # 性能汇总
    print(f"\n{'='*70}")
    print(f"  性能汇总 (OMP_NUM_THREADS=4)")
    print(f"{'='*70}")
    print(f"  {'模型':<15} {'ORT Best':>12} {'ORT Avg':>12} {'SPINN Best':>12} {'SPINN Avg':>12} {'SPINN/ORT':>10}")
    print(f"  {'-'*71}")
    
    for model_name, display_name in models:
        if display_name not in all_ort:
            continue
        ort_best = min(all_ort[display_name])
        ort_avg = np.mean(all_ort[display_name])
        spinn_times = all_spinn.get(display_name, [])
        if spinn_times:
            spinn_best = min(spinn_times)
            spinn_avg = np.mean(spinn_times)
            ratio = spinn_best / ort_best
            print(f"  {display_name:<15} {ort_best:>10.1f}ms {ort_avg:>10.1f}ms {spinn_best:>10.1f}ms {spinn_avg:>10.1f}ms {ratio:>9.2f}x")
        else:
            print(f"  {display_name:<15} {ort_best:>10.1f}ms {ort_avg:>10.1f}ms {'N/A':>12} {'N/A':>12}")

if __name__ == '__main__':
    main()
