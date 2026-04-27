#!/usr/bin/env python3
"""检查第一层Conv2D的参数"""

import onnx
import onnxruntime
import numpy as np
import onnx.numpy_helper

model = onnx.load("../complex_test.onnx")

# 找到第一个Conv节点
conv_node = model.graph.node[0]
print(f"第一层节点类型: {conv_node.op_type}")
print(f"输入: {list(conv_node.input)}")
print(f"输出: {list(conv_node.output)}")

print("\n属性:")
for attr in conv_node.attribute:
    if attr.name == "kernel_shape":
        print(f"  kernel_shape = {list(attr.ints)}")
    elif attr.name == "strides":
        print(f"  strides = {list(attr.ints)}")
    elif attr.name == "pads":
        print(f"  pads = {list(attr.ints)}")
    elif attr.name == "dilations":
        print(f"  dilations = {list(attr.ints)}")
    elif attr.name == "group":
        print(f"  group = {attr.i}")
    elif attr.name == "auto_pad":
        print(f"  auto_pad = {attr.s.decode() if attr.s else 'NOTSET'}")

# 获取权重
weights = {t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer}

for inp in conv_node.input:
    if inp in weights:
        w = weights[inp]
        print(f"\n权重 {inp}:")
        print(f"  shape = {w.shape}")
        print(f"  mean = {np.mean(w):.10f}")
        print(f"  std = {np.std(w):.10f}")
        
# 手动计算一个卷积输出查看
print("\n" + "=" * 80)
print("手动计算第一个输出位置 [0,0,0,0]")
print("=" * 80)

# 准备输入数据 (与 main.c 一致)
ort_session = onnxruntime.InferenceSession("../complex_test.onnx")
input_shape = ort_session.get_inputs()[0].shape
shape = [d if isinstance(d, int) else 1 for d in input_shape]
total = np.prod(shape)
input_data = np.array([i / 1000.0 for i in range(total)], dtype=np.float32).reshape(shape)

print(f"输入 shape: {input_data.shape}")
print(f"输入前几个值: {input_data.flatten()[:10]}")

# 获取权重
weight_name = conv_node.input[1]
kernel = weights[weight_name]
print(f"\n卷积核 shape: {kernel.shape}")  # [OC, IC, KH, KW]

# 手动计算 [0, 0, 0, 0] 位置 (N=0, OC=0, OH=0, OW=0)
# kernel shape: [16, 3, 3, 3]
# input shape: [1, 3, 32, 32]

OC = 0  # 第0个输出通道
N = 0
IC_count = kernel.shape[1]  # 3
KH, KW = kernel.shape[2], kernel.shape[3]  # 3, 3

# 获取padding参数
pads = [1, 1, 1, 1]  # default SAME padding for 3x3 kernel
for attr in conv_node.attribute:
    if attr.name == "pads":
        pads = list(attr.ints)

PH, PW = pads[0], pads[1]
print(f"Padding: PH={PH}, PW={PW}")

sum_val = 0.0
for ic in range(IC_count):
    for kh in range(KH):
        for kw in range(KW):
            ih = 0 - PH + kh  # OH=0, SH=1
            iw = 0 - PW + kw  # OW=0, SW=1
            
            if ih >= 0 and ih < input_data.shape[2] and iw >= 0 and iw < input_data.shape[3]:
                inp_val = input_data[N, ic, ih, iw]
                ker_val = kernel[OC, ic, kh, kw]
                sum_val += inp_val * ker_val
                if kh == 0 and kw < 3:
                    print(f"  kernel[{OC},{ic},{kh},{kw}] = {ker_val:.6f}, input[{N},{ic},{ih},{iw}] = {inp_val:.6f}")

print(f"\n手动计算的输出 [0,0,0,0] = {sum_val:.6f}")

# 运行ONNX获取真实值
outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: input_data})
onnx_out = outputs[0]
print(f"ONNX输出 [0,0,0,0] = {onnx_out[0,0,0,0]:.6f}")
print(f"ONNX输出 mean = {np.mean(onnx_out):.6f}")
