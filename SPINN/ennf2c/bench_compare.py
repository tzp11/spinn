#!/usr/bin/env python3
"""
性能/资源对比: SPINN runtime vs ennf2c generated vs ONNX Runtime

测量项:
- 推理时间 (best of N runs, with warmup)
- 二进制大小 (.text/.data/.rodata 段)
- 实测 RSS 峰值 (/usr/bin/time -v)
- Arena/权重大小 (静态)

用法:
    bench_compare.py <onnx_file> <ennf_file> <spinn_run_exe> <model_run_exe> [num_runs]
"""
import sys
import subprocess
import re
import os


def run_with_time(cmd):
    """Run command via /usr/bin/time -v, return (stdout, max_rss_kb)."""
    full = ["/usr/bin/time", "-v"] + cmd
    r = subprocess.run(full, capture_output=True, text=True)
    rss = None
    for line in r.stderr.splitlines():
        m = re.search(r"Maximum resident set size \(kbytes\):\s+(\d+)", line)
        if m:
            rss = int(m.group(1))
            break
    return r.stdout, r.stderr, rss


def parse_best_avg(stderr):
    best = avg = None
    for line in stderr.splitlines():
        m = re.search(r"Best:\s+([\d.]+)ms,\s+Avg:\s+([\d.]+)ms", line)
        if m:
            best = float(m.group(1))
            avg = float(m.group(2))
    return best, avg


def parse_run_only(stderr):
    """单次运行模式下的 'Run: Xms' 解析"""
    for line in stderr.splitlines():
        m = re.search(r"Run:\s+([\d.]+)ms", line)
        if m:
            return float(m.group(1))
    return None


def section_sizes(exe):
    """用 size 命令获取 text/data/bss 段大小"""
    try:
        r = subprocess.run(["size", exe], capture_output=True, text=True)
        lines = r.stdout.strip().splitlines()
        if len(lines) >= 2:
            parts = lines[1].split()
            return {
                "text": int(parts[0]),
                "data": int(parts[1]),
                "bss":  int(parts[2]),
                "total": int(parts[3]),
            }
    except Exception:
        pass
    return None


def file_size(path):
    return os.path.getsize(path) if os.path.exists(path) else 0


def fmt_kb(n):
    if n is None:
        return "n/a"
    if n >= 1024 * 1024:
        return f"{n/1024/1024:.2f} MB"
    if n >= 1024:
        return f"{n/1024:.1f} KB"
    return f"{n} B"


def bench_spinn(spinn_exe, ennf, num_runs):
    cmd = [spinn_exe, ennf, str(num_runs)]
    out, err, rss = run_with_time(cmd)
    best, avg = parse_best_avg(err)
    if best is None:
        # 单次模式 fallback
        best = parse_run_only(err)
        avg = best
    return best, avg, rss


def bench_generated(model_exe, num_runs):
    cmd = [model_exe, str(num_runs)]
    out, err, rss = run_with_time(cmd)
    best, avg = parse_best_avg(err)
    if best is None:
        best = parse_run_only(err)
        avg = best
    return best, avg, rss


def bench_ort(onnx_path, num_runs):
    """通过子进程跑 ORT, 测量 RSS 与时间"""
    script = f"""
import sys, time, numpy as np
import onnxruntime as ort
sess = ort.InferenceSession({onnx_path!r}, providers=['CPUExecutionProvider'])
inp = sess.get_inputs()[0]
shape = [d if isinstance(d, int) else 1 for d in inp.shape]
n = int(np.prod(shape))
data = (np.arange(n, dtype=np.float32) / 1000.0).reshape(shape)
# warmup
for _ in range(3):
    sess.run(None, {{inp.name: data}})
best = 1e18
total = 0.0
for _ in range({num_runs}):
    t = time.perf_counter()
    sess.run(None, {{inp.name: data}})
    e = (time.perf_counter() - t) * 1000.0
    if e < best: best = e
    total += e
print(f"Best: {{best:.2f}}ms, Avg: {{total/{num_runs}:.2f}}ms", file=sys.stderr)
"""
    cmd = [sys.executable, "-c", script]
    out, err, rss = run_with_time(cmd)
    best, avg = parse_best_avg(err)
    return best, avg, rss


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(1)
    onnx_path, ennf_path, spinn_exe, model_exe = sys.argv[1:5]
    num_runs = int(sys.argv[5]) if len(sys.argv) > 5 else 5

    print(f"== Benchmark (num_runs={num_runs}) ==\n")

    # ----- 二进制 / 模型大小 -----
    spinn_secs = section_sizes(spinn_exe)
    gen_secs = section_sizes(model_exe)
    ennf_size = file_size(ennf_path)
    onnx_size = file_size(onnx_path)
    onnx_data = file_size(onnx_path + ".data")
    onnx_total = onnx_size + onnx_data

    print("== File / Binary sizes ==")
    print(f"  ONNX file:        {fmt_kb(onnx_total)}  (.onnx={fmt_kb(onnx_size)} +.data={fmt_kb(onnx_data)})")
    print(f"  ENNF file:        {fmt_kb(ennf_size)}")
    print(f"  spinn_run binary: {fmt_kb(file_size(spinn_exe))}  (text={fmt_kb(spinn_secs['text']) if spinn_secs else 'n/a'})")
    print(f"  model_run binary: {fmt_kb(file_size(model_exe))}  (text={fmt_kb(gen_secs['text']) if gen_secs else 'n/a'}, "
          f"rodata in .data+.rodata)")
    print()

    # ----- 推理时间 -----
    print("== Inference time (best/avg) ==")
    print("  Running SPINN runtime...")
    s_best, s_avg, s_rss = bench_spinn(spinn_exe, ennf_path, num_runs)
    print(f"    best={s_best}ms  avg={s_avg}ms  RSS={fmt_kb((s_rss or 0)*1024)}")

    print("  Running ennf2c generated...")
    g_best, g_avg, g_rss = bench_generated(model_exe, num_runs)
    print(f"    best={g_best}ms  avg={g_avg}ms  RSS={fmt_kb((g_rss or 0)*1024)}")

    print("  Running ONNX Runtime...")
    o_best, o_avg, o_rss = bench_ort(onnx_path, num_runs)
    print(f"    best={o_best}ms  avg={o_avg}ms  RSS={fmt_kb((o_rss or 0)*1024)}")
    print()

    # ----- 汇总 -----
    print("== Summary ==")
    print(f"  {'Backend':<14} {'best(ms)':>10} {'avg(ms)':>10} {'RSS':>14} {'binary':>14}")
    def row(name, best, avg, rss, binsz):
        b = f"{best:.2f}" if best is not None else "n/a"
        a = f"{avg:.2f}" if avg is not None else "n/a"
        r = fmt_kb((rss or 0)*1024)
        print(f"  {name:<14} {b:>10} {a:>10} {r:>14} {binsz:>14}")
    row("SPINN runtime", s_best, s_avg, s_rss, fmt_kb(file_size(spinn_exe)))
    row("ennf2c gen",    g_best, g_avg, g_rss, fmt_kb(file_size(model_exe)))
    row("ONNX Runtime",  o_best, o_avg, o_rss, "N/A (Python)")

    if s_best and g_best:
        ratio = g_best / s_best
        delta = g_best - s_best
        print(f"\n  Time(gen/runtime) = {ratio:.3f}x  (delta = {delta:+.2f}ms)")
    if s_rss and g_rss:
        rratio = g_rss / s_rss
        rdelta = (g_rss - s_rss) * 1024
        print(f"  RSS(gen/runtime)  = {rratio:.3f}x  (delta = {fmt_kb(rdelta)})")


if __name__ == "__main__":
    main()
