
import onnxruntime
import numpy as np

# 1. 获取 ONNX 最终输出
ort = onnxruntime.InferenceSession('../complex_test.onnx')
shape = [1, 3, 32, 32]
data = np.array([i / 1000.0 for i in range(np.prod(shape))], dtype=np.float32).reshape(shape)
onnx_out = ort.run(None, {'input': data})[0]

# 2. 读取 SPINN 最终输出 (从 spinn_trace.txt 尾部提取 Result: 之后的一行)
spinn_out_list = []
with open("spinn_trace.txt", "r") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip() == "Result:":
            if i + 1 < len(lines):
                vals = lines[i+1].strip().split()
                spinn_out_list = [float(x) for x in vals]
            break

if not spinn_out_list:
    print("❌ Failed to parse SPINN output from spinn_trace.txt")
    exit(1)

spinn_out = np.array(spinn_out_list, dtype=np.float32)

print(f"\nFinal Layer Comparison:")
print(f"  Shape: ONNX {onnx_out.shape} vs SPINN {spinn_out.shape}")
print(f"  ONNX Output: {onnx_out.flatten()[:10]}")
print(f"  SPINN Output: {spinn_out[:10]}")

# 3. 对比
diff = np.abs(onnx_out.flatten() - spinn_out)
max_diff = np.max(diff)
mean_diff = np.mean(diff)

print(f"\nMax Diff: {max_diff:.6f}")
print(f"Mean Diff: {mean_diff:.6f}")

if max_diff < 1e-4:
    print("\n✅ Final Result MATCHES!")
else:
    print("\n❌ Final Result MISMATCH!")
