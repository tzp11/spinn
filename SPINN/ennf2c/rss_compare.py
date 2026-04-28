#!/usr/bin/env python3
"""
RSS 同口径对比: SPINN runtime / ennf2c-generated / ORT

口径统一原则:
- 都用子进程方式启动, 用 /usr/bin/time -v 抓 OS 级 Maximum RSS
- 都跑相同 num_runs 次推理 (固定 input)
- 都使用相同的 OMP_NUM_THREADS=4

并额外测量 RSS 增量分解:
- 启动 baseline   (空程序刚 exec 后立即 sleep 测 RSS)
- 推理后 peak     (跑完推理后 max RSS)
- 应用增量        = peak - baseline

对 ORT 由于 Python 解释器开销, baseline 自动包含 Python+import 开销.
对 SPINN runtime 是 C 二进制, baseline 极小.

用法:
    rss_compare.py <onnx_file> <ennf_file> <spinn_exe> <model_run_exe> [num_runs]
"""
import os
import re
import sys
import subprocess
import tempfile

NUM_RUNS_DEFAULT = 10


def time_v(cmd, env=None):
    """用 /usr/bin/time -v 跑 cmd, 返回 (max_rss_kb, stderr_text)."""
    full = ["/usr/bin/time", "-v"] + cmd
    e = os.environ.copy()
    e.setdefault("OMP_NUM_THREADS", "4")
    if env:
        e.update(env)
    r = subprocess.run(full, capture_output=True, text=True, env=e)
    rss = None
    for line in r.stderr.splitlines():
        m = re.search(r"Maximum resident set size \(kbytes\):\s+(\d+)", line)
        if m:
            rss = int(m.group(1))
            break
    return rss, r.stdout, r.stderr


def baseline_python_ort():
    """测量 'import onnxruntime' 后立即退出的 Python 进程 RSS (作为 ORT baseline)."""
    code = """
import onnxruntime, sys, os
# 不创建 session, 仅 import
sys.exit(0)
"""
    rss, _, _ = time_v([sys.executable, "-c", code])
    return rss


def measure_ort_full(onnx_path, num_runs):
    """完整跑 ORT 推理: baseline 已 import + load + N 次 run."""
    code = f"""
import onnxruntime as ort
import numpy as np
sess = ort.InferenceSession({onnx_path!r}, providers=['CPUExecutionProvider'])
inp = sess.get_inputs()[0]
shape = [d if isinstance(d, int) else 1 for d in inp.shape]
n = int(np.prod(shape))
data = (np.arange(n, dtype=np.float32) / 1000.0).reshape(shape)
# warmup
for _ in range(3):
    sess.run(None, {{inp.name: data}})
for _ in range({num_runs}):
    sess.run(None, {{inp.name: data}})
"""
    rss, _, _ = time_v([sys.executable, "-c", code])
    return rss


def baseline_c_min():
    """测量最小 C 进程的 baseline (作为 SPINN/generated baseline)."""
    rss, _, _ = time_v(["/bin/true"])
    return rss


def measure_spinn(spinn_exe, ennf_path, num_runs):
    rss, _, _ = time_v([spinn_exe, ennf_path, str(num_runs)])
    return rss


def measure_generated(model_exe, num_runs):
    rss, _, _ = time_v([model_exe, str(num_runs)])
    return rss


def fmt_kb(n):
    if n is None:
        return "n/a"
    if n >= 1024 * 1024:
        return f"{n/1024/1024:.2f} GB"
    if n >= 1024:
        return f"{n/1024:.2f} MB"
    return f"{n} KB"


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(1)
    onnx, ennf, spinn_exe, gen_exe = sys.argv[1:5]
    num_runs = int(sys.argv[5]) if len(sys.argv) > 5 else NUM_RUNS_DEFAULT

    print(f"== RSS Same-Methodology Comparison (num_runs={num_runs}, OMP=4) ==\n")

    # baselines
    print("Measuring baselines...")
    py_base = baseline_python_ort()
    c_base = baseline_c_min()
    print(f"  Python+ORT import baseline: {fmt_kb(py_base)}")
    print(f"  Minimal C process baseline: {fmt_kb(c_base)}")
    print()

    # full
    print("Measuring full inference RSS (load + N runs)...")
    s_rss = measure_spinn(spinn_exe, ennf, num_runs)
    g_rss = measure_generated(gen_exe, num_runs)
    o_rss = measure_ort_full(onnx, num_runs)

    print()
    print("== OS-level Max RSS (/usr/bin/time -v) ==")
    print(f"  {'Backend':<22} {'Total RSS':>14} {'-baseline':>14} {'(baseline)':>14}")
    print(f"  {'-'*22} {'-'*14} {'-'*14} {'-'*14}")
    print(f"  {'SPINN runtime (C)':<22} {fmt_kb(s_rss):>14} "
          f"{fmt_kb(s_rss - c_base):>14} {fmt_kb(c_base):>14}")
    print(f"  {'ennf2c generated (C)':<22} {fmt_kb(g_rss):>14} "
          f"{fmt_kb(g_rss - c_base):>14} {fmt_kb(c_base):>14}")
    print(f"  {'ORT (Python)':<22} {fmt_kb(o_rss):>14} "
          f"{fmt_kb(o_rss - py_base):>14} {fmt_kb(py_base):>14}")
    print()

    print("== 解读 ==")
    print(f"  Total RSS:          整个进程 OS 级 RSS, 含解释器/共享库/线程栈/heap")
    print(f"  -baseline (incr):   减去启动基线后的 'app 增量'")
    print(f"                      用于评估模型本身的内存代价 (跨语言公平)")

    if s_rss and o_rss:
        s_incr = s_rss - c_base
        o_incr = o_rss - py_base
        ratio = o_incr / s_incr if s_incr else float("inf")
        print(f"\n  ORT/SPINN incr ratio = {ratio:.2f}x  (越大说明 SPINN 越省)")

if __name__ == "__main__":
    main()
