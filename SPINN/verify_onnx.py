import onnx
import onnxruntime
import numpy as np

import sys
import time

model_path = sys.argv[1] if len(sys.argv) > 1 else "complex_test.onnx"
num_threads = int(sys.argv[2]) if len(sys.argv) > 2 else 4

print(f"Loading {model_path} with {num_threads} threads...")

sess_options = onnxruntime.SessionOptions()
sess_options.intra_op_num_threads = num_threads
sess_options.inter_op_num_threads = 1
session = onnxruntime.InferenceSession(model_path, sess_options)

input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape

# Resolve dynamic shape logic for SPINN compatibility
shape = list(input_shape)
for i in range(len(shape)):
    if type(shape[i]) is str or shape[i] is None:
        shape[i] = 1 # Assume batch 1

print(f"Input: {input_name}, Shape: {shape}")

# Generate identical SPINN input (val = i / 1000.0f)
total_elements = np.prod(shape)
dummy_input = np.arange(total_elements, dtype=np.float32) / 1000.0
dummy_input = dummy_input.reshape(shape)

# Warmup
for _ in range(5):
    outputs = session.run(None, {input_name: dummy_input})

# Benchmark
runs = 10
start = time.time()
for _ in range(runs):
    outputs = session.run(None, {input_name: dummy_input})
end = time.time()

avg_ms = (end - start) * 1000 / runs
print(f"ORT Inference ({num_threads}T) Avg Time: {avg_ms:.2f} ms")

out_flat = outputs[0].flatten()
out_str = " ".join([f"{x:.6f}" for x in out_flat[:10]])
print(f"ORT Result: {out_str}")

