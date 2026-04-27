import onnxruntime as ort
import numpy as np

# 加载ONNX模型
sess = ort.InferenceSession("../mnist.simplified.onnx")

# 准备输入：全 1 的张量，形状 [1, 1, 14, 14]
# float32 类型
input_data = np.ones((1, 1, 14, 14), dtype=np.float32)

# 推理
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
result = sess.run([output_name], {input_name: input_data})[0]

# 打印结果
print("Reference Output (ONNX Runtime):")
print(result)

# 打印权重的参考值
print("\nReference Weights:")

import onnx
from onnx import numpy_helper
model = onnx.load("../mnist.simplified.onnx")
for init in model.graph.initializer:
    w = numpy_helper.to_array(init).flatten()
    print(f"{init.name}: {w[0]:.6f} {w[1]:.6f} {w[2]:.6f} {w[3]:.6f} {w[4]:.6f}")

# 打印我们的 TinySPINN 结果进行对比
print("\nTinySPINN Output:")
print("-0.351000 0.107895 0.100940 0.011734 -0.111144 0.418509 -0.152223 0.008702 -0.104102 0.075627")
