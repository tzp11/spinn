#!/usr/bin/env python3
"""详细检查LayerNormalization的输入输出，确定问题来源"""

import onnx
import onnxruntime
import numpy as np
import onnx.numpy_helper
import struct

print("=" * 80)
print("LayerNormalization 问题诊断")
print("=" * 80)

# ========================
# 1. 检查 ONNX 模型
# ========================
print("\n【1】ONNX 模型检查")
model = onnx.load("../complex_test.onnx")

ln_node = None
for i, node in enumerate(model.graph.node):
    if node.op_type == "LayerNormalization":
        ln_node = node
        print(f"  LayerNormalization 是第 {i} 个节点")
        break

print(f"  输入: {list(ln_node.input)}")
print(f"  输出: {list(ln_node.output)}")

# 检查属性
epsilon = 1e-5
axis = -1
for attr in ln_node.attribute:
    if attr.name == "epsilon":
        epsilon = attr.f
        print(f"  epsilon = {epsilon}")
    elif attr.name == "axis":
        axis = attr.i
        print(f"  axis = {axis}")

# ========================
# 2. 检查 ENNF 转换器输出
# ========================
print("\n【2】ENNF 转换器检查")

# LayerNorm params 结构体
# typedef struct {
#     float    epsilon;     // 4 bytes
#     int32_t  axis;        // 4 bytes
#     uint8_t  stash_type;  // 1 byte
#     uint8_t  reserved[3]; // 3 bytes
#     int32_t  num_groups;  // 4 bytes (GroupNorm only)
# } ENNF_LayerNormParams;

with open("../complex_test.ennf", "rb") as f:
    header = f.read(64)
    num_nodes = struct.unpack("<I", header[8:12])[0]
    node_table_offset = struct.unpack("<I", header[36:40])[0]
    
    f.seek(node_table_offset)
    
    for i in range(num_nodes):
        # ENNF_NodeBase is 8 bytes fixed header
        base = f.read(8)
        op_type, num_inputs, num_outputs, params_size, flags, reserved = struct.unpack("<HBBHBB", base)
        
        # 读取input/output ids
        input_ids = struct.unpack(f"<{num_inputs}H", f.read(num_inputs * 2))
        output_ids = struct.unpack(f"<{num_outputs}H", f.read(num_outputs * 2))
        
        # 读取params
        params = f.read(params_size) if params_size > 0 else b""
        
        if op_type == 103:  # LayerNorm
            print(f"  Node {i}: op_type=103 (LayerNormalization)")
            print(f"    inputs: {input_ids}, outputs: {output_ids}")
            print(f"    params_size: {params_size} bytes")
            
            if params_size >= 8:
                eps_ennf, axis_ennf = struct.unpack("<fi", params[:8])
                print(f"    ENNF epsilon: {eps_ennf}")
                print(f"    ENNF axis: {axis_ennf}")
                
                if abs(eps_ennf - epsilon) > 1e-10:
                    print(f"    ⚠️  EPSILON 不匹配!")
                if axis_ennf != axis:
                    print(f"    ⚠️  AXIS 不匹配!")
            break

# ========================
# 3. 检查 SPINN 推理
# ========================
print("\n【3】SPINN 推理对比")

# 运行ONNX获取中间结果
for node in model.graph.node:
    for out in node.output:
        if out not in {o.name for o in model.graph.output}:
            vi = onnx.helper.make_tensor_value_info(out, onnx.TensorProto.FLOAT, None)
            model.graph.output.append(vi)
onnx.save(model, "debug_tmp.onnx")

ort = onnxruntime.InferenceSession("debug_tmp.onnx")
shape = [1, 3, 32, 32]
data = np.array([i / 1000.0 for i in range(np.prod(shape))], dtype=np.float32).reshape(shape)
out_names = [o.name for o in ort.get_outputs()]
outputs = ort.run(out_names, {"input": data})
out_map = {n: v for n, v in zip(out_names, outputs)}

# 获取LayerNorm的输入和输出
ln_input_name = ln_node.input[0]
ln_output_name = ln_node.output[0]

ln_input = out_map.get(ln_input_name)
ln_output = out_map.get(ln_output_name)

print(f"  LayerNorm 输入 ({ln_input_name}):")
print(f"    shape: {ln_input.shape}")
print(f"    mean: {np.mean(ln_input):.6f}")
print(f"    val[0:3]: {ln_input.flatten()[:3]}")

print(f"\n  LayerNorm ONNX输出 ({ln_output_name}):")
print(f"    shape: {ln_output.shape}")
print(f"    mean: {np.mean(ln_output):.6f}")
print(f"    val[0:3]: {ln_output.flatten()[:3]}")

# 读取SPINN trace
print(f"\n  从 spinn_trace.txt 查看SPINN结果:")
with open("spinn_trace.txt") as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if "Node 8" in line:
            for j in range(i, min(i+6, len(lines))):
                print(f"    {lines[j].rstrip()}")
            break

# ========================
# 4. 手动计算对比
# ========================
print("\n【4】手动计算验证")

weights = {t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer}
scale = weights.get(ln_node.input[1], np.ones(16))
bias = weights.get(ln_node.input[2], np.zeros(16))

# LayerNorm: 对最后一个维度进行归一化
# 输入: [1, 256, 16], 对 axis=-1 (最后16个元素) 归一化
input_2d = ln_input.reshape(-1, ln_input.shape[-1])  # [256, 16]

output_manual = np.zeros_like(input_2d)
for i in range(input_2d.shape[0]):
    row = input_2d[i]
    mean = np.mean(row)
    var = np.var(row)
    normalized = (row - mean) / np.sqrt(var + epsilon)
    output_manual[i] = normalized * scale + bias

output_manual = output_manual.reshape(ln_input.shape)

print(f"  手动计算结果:")
print(f"    mean: {np.mean(output_manual):.6f}")
print(f"    val[0:3]: {output_manual.flatten()[:3]}")

diff = np.abs(output_manual - ln_output)
print(f"\n  与ONNX差异:")
print(f"    max diff: {np.max(diff):.10f}")

if np.max(diff) < 1e-5:
    print("  ✅ 手动计算与ONNX完全匹配")
else:
    print("  ❌ 计算有差异")

# ========================
# 5. 结论
# ========================
print("\n" + "=" * 80)
print("诊断结论")
print("=" * 80)
