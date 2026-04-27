#!/usr/bin/env python3
"""专门诊断 LayerNormalization 层的问题"""

import onnx
import onnxruntime  
import numpy as np

# Load model
model = onnx.load("../complex_test.onnx")

# 提取 LayerNormalization 节点
ln_node = None
for node in model.graph.node:
    if node.op_type == "LayerNormalization":
        ln_node = node
        break

if not ln_node:
    print("No LayerNormalization node found!")
    exit(1)

print("=" * 80)
print(f"Layer Normalization 节点分析")
print("=" * 80)
print(f"输入: {ln_node.input}")
print(f"输出: {ln_node.output}")

# 检查属性
print(f"\n属性:")
for attr in ln_node.attribute:
    if attr.name == "axis":
        print(f"  axis = {attr.i}")
    elif attr.name == "epsilon":
        print(f"  epsilon = {attr.f}")
    elif attr.name == "stash_type":
        print(f"  stash_type = {attr.i}")

# 获取权重
weights = {t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer}

for inp in ln_node.input:
    if inp in weights:
        w = weights[inp]
        print(f"\n权重 {inp}:")
        print(f"  shape = {w.shape}")
        print(f"  mean = {np.mean(w):.6f}")
        print(f"  std = {np.std(w):.6f}")
        print(f"  values = {w.flatten()[:10]}")

# 运行完整模型获取这一层的输入输出
ort_session = onnxruntime.InferenceSession("../complex_test.onnx")
input_name = ort_session.get_inputs()[0].name
input_shape = ort_session.get_inputs()[0].shape
shape = [d if isinstance(d, int) else 1 for d in input_shape]
total = np.prod(shape)
data = np.array([i / 1000.0 for i in range(total)], dtype=np.float32).reshape(shape)

# 添加中间输出
for node in model.graph.node:
    for output in node.output:
        if output not in {o.name for o in model.graph.output}:
            vi = onnx.helper.make_tensor_value_info(output, onnx.TensorProto.FLOAT, None)
            model.graph.output.append(vi)

onnx.save(model, "debug_tmp.onnx")
ort_session = onnxruntime.InferenceSession("debug_tmp.onnx")
sess_out_names = [o.name for o in ort_session.get_outputs()]
outputs = ort_session.run(sess_out_names, {input_name: data})
out_map = {name: val for name, val in zip(sess_out_names, outputs)}

# 找到这一层的输入输出
ln_input_data = None
ln_output_data = None

for inp in ln_node.input:
    if inp in out_map:
        ln_input_data = out_map[inp]
        print(f"\n实际输入 {inp}:")
        print(f"  shape = {ln_input_data.shape}")
        print(f"  mean = {np.mean(ln_input_data):.6f}")
        print(f"  std = {np.std(ln_input_data):.6f}")
        print(f"  first 10 values = {ln_input_data.flatten()[:10]}")

for out in ln_node.output:
    if out in out_map:
        ln_output_data = out_map[out]
        print(f"\n实际输出 {out}:")
        print(f"  shape = {ln_output_data.shape}")
        print(f"  mean = {np.mean(ln_output_data):.6f}")
        print(f"  std = {np.std(ln_output_data):.6f}")
        print(f"  first 10 values = {ln_output_data.flatten()[:10]}")

# 手动计算 LayerNorm
if ln_input_data is not None:
    print("\n" + "=" * 80)
    print("手动计算 LayerNorm (用于验证)")
    print("=" * 80)
    
    # 假设 axis=-1 (最后一个维度)
    # 输入shape: [1, 256, 16]
    # 对最后维度(16)进行normalization
    
    eps = 1e-5
    scale = weights.get(ln_node.input[1], np.ones(16))
    bias = weights.get(ln_node.input[2], np.zeros(16))
    
    # Reshape to [1*256, 16]
    input_2d = ln_input_data.reshape(-1, ln_input_data.shape[-1])
    print(f"Reshaped input: {input_2d.shape}")
    
    # 逐instance计算
    output_manual = np.zeros_like(input_2d)
    for i in range(input_2d.shape[0]):
        row = input_2d[i]
        mean = np.mean(row)
        var = np.var(row)
        normalized = (row - mean) / np.sqrt(var + eps)
        output_manual[i] = normalized * scale + bias
    
    output_manual = output_manual.reshape(ln_input_data.shape)
    
    print(f"手动计算结果:")
    print(f"  mean = {np.mean(output_manual):.6f}")
    print(f"  std = {np.std(output_manual):.6f}")
    print(f"  first 10 values = {output_manual.flatten()[:10]}")
    
    if ln_output_data is not None:
        diff = np.abs(output_manual - ln_output_data)
        print(f"\n与ONNX输出的差异:")
        print(f"  max diff = {np.max(diff):.6f}")
        print(f"  mean diff = {np.mean(diff):.6f}")
