
import onnxruntime
import numpy as np

try:
    ort = onnxruntime.InferenceSession('resnet101.onnx')
except Exception as e:
    print(f"Error loading ONNX: {e}")
    exit(1)

input_name = ort.get_inputs()[0].name
shape = ort.get_inputs()[0].shape

count = np.prod(shape)
# C input: input_data[i] = (float)i / 1000.0f;
data_flat = np.array([i / 1000.0 for i in range(count)], dtype=np.float32)
data = data_flat.reshape(shape)

onnx_out = ort.run(None, {input_name: data})[0]

print("ONNX Result:  ", end="")
flat = onnx_out.flatten()
for i in range(10):
    print(f"{flat[i]:.6f} ", end="")
print("")
