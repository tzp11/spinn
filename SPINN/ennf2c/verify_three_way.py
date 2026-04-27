#!/usr/bin/env python3
"""
三方对比验证脚本: ORT vs SPINN runtime vs ennf2c generated C

用法:
    verify_three_way.py <onnx_file> <ennf_file> <spinn_run_exe> <model_run_exe>

输入: 与 SPINN main.c / ennf2c main_test.c 一致
    input[i] = i / 1000.0  (float32)
"""
import sys
import subprocess
import numpy as np


def run_ort(onnx_path):
    """运行 ONNX Runtime"""
    import onnxruntime
    sess = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    n = int(np.prod(shape))
    data = np.arange(n, dtype=np.float32) / 1000.0
    data = data.reshape(shape)
    out = sess.run(None, {inp.name: data})[0]
    return out.flatten()


def parse_result_line(stdout, prefix):
    """从 stdout 中找以 prefix 开头的行, 解析浮点列表"""
    for line in stdout.splitlines():
        if line.startswith(prefix):
            tail = line[len(prefix):].strip()
            try:
                return np.array([float(x) for x in tail.split()])
            except ValueError:
                continue
    return None


def run_spinn(spinn_exe, ennf_path):
    """运行 SPINN runtime, 输出格式: 'SPINN Result: ...'"""
    r = subprocess.run([spinn_exe, ennf_path], capture_output=True, text=True)
    return parse_result_line(r.stdout, "SPINN Result:")


def run_generated(model_run_exe):
    """运行 ennf2c 生成的可执行文件, 输出格式: 'Generated Result: ...'"""
    r = subprocess.run([model_run_exe], capture_output=True, text=True)
    return parse_result_line(r.stdout, "Generated Result:")


def report(name_a, vals_a, name_b, vals_b):
    """对比两个数组: 同时检查绝对误差和相对误差。

    PASS 条件: max_diff < 1e-3  OR  max_rel_diff < 1e-5
    （应对大数值输出场景, 例如 ResNet 输出可达 1e6 量级时绝对误差大但相对误差小）
    """
    if vals_a is None or vals_b is None:
        print(f"  [{name_a} vs {name_b}]  SKIP (missing data)")
        return False
    n = min(len(vals_a), len(vals_b))
    a = vals_a[:n].astype(np.float64)
    b = vals_b[:n].astype(np.float64)
    diff = np.abs(a - b)
    max_diff = float(diff.max()) if n > 0 else float("inf")
    mean_diff = float(diff.mean()) if n > 0 else float("inf")
    denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-12)
    rel = diff / denom
    max_rel = float(rel.max()) if n > 0 else float("inf")
    ok = (max_diff < 1e-3) or (max_rel < 1e-5)
    status = "PASS" if ok else "FAIL"
    print(f"  [{name_a} vs {name_b}]  n={n}  max_abs={max_diff:.3e}  "
          f"max_rel={max_rel:.3e}  mean_abs={mean_diff:.3e}  {status}")
    return ok


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(1)
    onnx_path, ennf_path, spinn_exe, model_exe = sys.argv[1:5]

    print(f"== Three-way verify ==")
    print(f"  ONNX:      {onnx_path}")
    print(f"  ENNF:      {ennf_path}")
    print(f"  SPINN:     {spinn_exe}")
    print(f"  Generated: {model_exe}")
    print()

    ort_out = run_ort(onnx_path)
    spinn_out = run_spinn(spinn_exe, ennf_path)
    gen_out = run_generated(model_exe)

    n_show = 10
    def head(arr):
        if arr is None:
            return "None"
        return " ".join(f"{x:.6f}" for x in arr[:n_show])

    print(f"  ORT       (first {n_show}): {head(ort_out)}")
    print(f"  SPINN     (first {n_show}): {head(spinn_out)}")
    print(f"  Generated (first {n_show}): {head(gen_out)}")
    print()

    print("== Pairwise diffs ==")
    p1 = report("SPINN    ", spinn_out, "Generated", gen_out)
    p2 = report("ORT      ", ort_out,   "SPINN    ", spinn_out)
    p3 = report("ORT      ", ort_out,   "Generated", gen_out)

    print()
    if p1 and p2 and p3:
        print("OVERALL: PASS")
        sys.exit(0)
    elif p1:
        print("OVERALL: SPINN==Generated (good), but disagree with ORT (algo difference?)")
        sys.exit(2)
    else:
        print("OVERALL: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
