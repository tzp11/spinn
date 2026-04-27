#!/usr/bin/env python3
"""
profile_ort.py - ONNX Runtime 内存占用分析工具
对比 SPINN 的 profile_memory 输出
"""

import os
import sys
import tracemalloc
import resource
import onnxruntime as ort
import numpy as np

def get_rss_kb():
    """获取当前进程 RSS (Resident Set Size) in KB"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

def profile_model(onnx_path, ext_data_path=None):
    name = os.path.basename(onnx_path)
    print("=" * 60)
    print(f"模型: {name}")
    print(f"ONNX Runtime 版本: {ort.__version__}")
    print("=" * 60)

    # 文件大小
    onnx_size = os.path.getsize(onnx_path)
    ext_size = os.path.getsize(ext_data_path) if ext_data_path and os.path.exists(ext_data_path) else 0
    print(f"ONNX 文件: {onnx_size:,} bytes + ext_data: {ext_size:,} bytes")
    print()

    # 基线内存
    rss_before = get_rss_kb()
    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()

    # 创建 session
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.enable_mem_pattern = True
    opts.enable_cpu_mem_arena = True
    # 禁用所有优化以获取最纯净的对比
    # opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    print("--- 加载模型 (创建 InferenceSession) ---")
    session = ort.InferenceSession(onnx_path, opts, providers=['CPUExecutionProvider'])

    rss_after_load = get_rss_kb()
    snap_after_load = tracemalloc.take_snapshot()

    # 获取输入输出信息
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    print(f"  输入: {[(i.name, i.shape, i.type) for i in inputs]}")
    print(f"  输出: {[(o.name, o.shape, o.type) for o in outputs]}")

    # 构造输入
    feed = {}
    for inp in inputs:
        shape = []
        for d in inp.shape:
            if isinstance(d, str) or d is None:
                shape.append(1)  # dynamic dim -> 1
            else:
                shape.append(d)
        if 'float' in inp.type.lower() or inp.type == 'tensor(float)':
            feed[inp.name] = np.random.randn(*shape).astype(np.float32)
        elif 'int64' in inp.type.lower():
            feed[inp.name] = np.zeros(shape, dtype=np.int64)
        else:
            feed[inp.name] = np.random.randn(*shape).astype(np.float32)

    # 推理
    print("\n--- 执行推理 ---")
    result = session.run(None, feed)

    rss_after_run = get_rss_kb()
    snap_after_run = tracemalloc.take_snapshot()

    tracemalloc.stop()

    # 计算 Python tracemalloc 统计
    stats_load = snap_after_load.compare_to(snap_before, 'filename')
    stats_run = snap_after_run.compare_to(snap_before, 'filename')

    total_load = sum(s.size for s in snap_after_load.statistics('filename'))
    total_run = sum(s.size for s in snap_after_run.statistics('filename'))

    print()
    print("--- ORT 内存占用 ---")
    print()
    print(f"  [RSS (操作系统层面, 含共享库)]")
    print(f"    基线 (import 后):        {rss_before:>10,} KB")
    print(f"    加载模型后:              {rss_after_load:>10,} KB  (+{rss_after_load - rss_before:,} KB)")
    print(f"    推理一次后:              {rss_after_run:>10,} KB  (+{rss_after_run - rss_before:,} KB)")
    print()
    print(f"  [Python tracemalloc (Python堆)]")
    print(f"    加载后 Python 堆:        {total_load:>10,} bytes ({total_load/1024:.1f} KB)")
    print(f"    推理后 Python 堆:        {total_run:>10,} bytes ({total_run/1024:.1f} KB)")
    print()

    # ORT 自带的内存统计（如果可用）
    try:
        mem_info = session.get_session_options()
        print(f"  [ORT Session 配置]")
        print(f"    enable_mem_pattern:       {opts.enable_mem_pattern}")
        print(f"    enable_cpu_mem_arena:     {opts.enable_cpu_mem_arena}")
        print(f"    intra_op_num_threads:     {opts.intra_op_num_threads}")
    except:
        pass

    # 输出结果前几个值
    print()
    print(f"  [输出结果 (前10值)]")
    for i, out in enumerate(result):
        flat = out.flatten()
        n = min(10, len(flat))
        vals = " ".join(f"{v:.6f}" for v in flat[:n])
        print(f"    output[{i}] shape={out.shape}: {vals}")

    print()
    print(f"  ====================================")
    print(f"  ORT 总 RSS 增量: {rss_after_run - rss_before:,} KB ({(rss_after_run - rss_before)/1024:.2f} MB)")
    print(f"  ====================================")
    print()

    return {
        'name': name,
        'rss_baseline': rss_before,
        'rss_after_load': rss_after_load,
        'rss_after_run': rss_after_run,
        'rss_delta': rss_after_run - rss_before,
    }


if __name__ == '__main__':
    models = [
        ('/home/tzp/work/SPINN/SPINN/mnist.simplified.onnx', None),
        ('/home/tzp/work/SPINN/SPINN/run_time/yolov10n.onnx', None),
        ('/home/tzp/work/SPINN/SPINN/run_time/resnet101.onnx',
         '/home/tzp/work/SPINN/SPINN/run_time/resnet101.onnx.data'),
    ]

    results = []
    for onnx_path, ext_data in models:
        if not os.path.exists(onnx_path):
            print(f"SKIP: {onnx_path} not found")
            continue
        try:
            r = profile_model(onnx_path, ext_data)
            results.append(r)
        except Exception as e:
            print(f"ERROR on {onnx_path}: {e}")
            import traceback
            traceback.print_exc()
        print()

    # 汇总对比表
    if results:
        print("=" * 60)
        print("汇总：ORT RSS 增量 vs SPINN 总内存")
        print("=" * 60)
        spinn_data = {
            'mnist.simplified.onnx': 19188,
            'yolov10n.onnx': 33934536,
            'resnet101.onnx': 189861992,
        }
        print(f"{'模型':<25} {'ORT RSS增量':>15} {'SPINN总内存':>15} {'ORT/SPINN':>10}")
        for r in results:
            ort_bytes = r['rss_delta'] * 1024
            spinn_bytes = spinn_data.get(r['name'], 0)
            ratio = ort_bytes / spinn_bytes if spinn_bytes > 0 else 0
            print(f"{r['name']:<25} {ort_bytes:>12,} B {spinn_bytes:>12,} B {ratio:>9.1f}x")
