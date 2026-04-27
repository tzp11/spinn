import onnx
import onnxruntime
import numpy as np
import sys
import struct

def run_onnx(onnx_path, input_val):
    ort_session = onnxruntime.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    # 假设输入 shape 是 [1, 128, 64]
    input_shape = ort_session.get_inputs()[0].shape
    # 处理可能的动态 shape
    shape = [d if isinstance(d, int) else 1 for d in input_shape]
    
    # 构造输入数据 (与 main.c 中的逻辑一致: i / 1000.0)
    total_elems = 1
    for d in shape: total_elems *= d
    
    data = np.array([i / 1000.0 for i in range(total_elems)], dtype=np.float32).reshape(shape)
    
    outputs = ort_session.run(None, {input_name: data})
    return outputs[0].flatten()

def run_spinn(spinn_exe, ennf_path):
    import subprocess
    result = subprocess.run([spinn_exe, ennf_path], capture_output=True, text=True)
    
    # 解析输出: Result:\n0.015940 ...
    lines = result.stdout.split("\n")
    try:
        idx = lines.index("Result:")
        vals = [float(x) for x in lines[idx+1].strip().split()]
        return np.array(vals)
    except:
        print("SPINN output parse error:", result.stdout)
        return None

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: verify.py <onnx_file> <ennf_file> [spinn_exe]")
        sys.exit(1)
        
    onnx_file = sys.argv[1]
    ennf_file = sys.argv[2]
    spinn_exe = sys.argv[3] if len(sys.argv) > 3 else "./spinn_run"
    
    print(f"Running ONNX: {onnx_file}")
    ref_out = run_onnx(onnx_file, 0)
    print("Reference output (first 10):", ref_out[:10])
    
    print(f"Running SPINN: {ennf_file}")
    spinn_out = run_spinn(spinn_exe, ennf_file)
    if spinn_out is None: sys.exit(1)
    print("SPINN output:", spinn_out)
    
    # 对比 (只对比前 10 个，因为 main.c 只打印了 10 个)
    n = min(len(ref_out), len(spinn_out))
    diff = np.abs(ref_out[:n] - spinn_out[:n])
    max_diff = np.max(diff)
    
    print(f"Max Diff (first {n}): {max_diff}")
    if max_diff < 1e-4:
        print("PASS")
    else:
        print("FAIL")

