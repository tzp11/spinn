#!/usr/bin/env python3
"""详细对比前几层，找到最初的偏差点"""

import re

with open("onnx_trace.txt") as f:
    onnx_lines = f.readlines()
with open("spinn_trace.txt") as f:
    spinn_lines = f.readlines()

print("=" * 100)
print("逐层详细对比 (前10层)")
print("=" * 100)

# 提取所有层的输出统计
def extract_outputs(lines):
    """提取每一层的输出统计"""
    layers = {}
    current_layer = -1
    
    for line in lines:
        # 检测节点开始
        match = re.search(r'Node (\d+)', line)
        if match:
            current_layer = int(match.group(1))
            layers[current_layer] = {'inputs': [], 'outputs': []}
            continue
        
        # 提取输出统计
        if current_layer >= 0 and 'Out' in line and 'mean=' in line:
            match = re.search(r'mean=([-\deE.+]+).*?val=\[([-\deE.+]+)\s+([-\deE.+]+)', line)
            if match:
                try:
                    layers[current_layer]['outputs'].append({
                        'mean': float(match.group(1)),
                        'val0': float(match.group(2)),
                        'val1': float(match.group(3))
                    })
                except ValueError:
                    pass  # Skip invalid values like -nan
        
        # 提取输入统计
        if current_layer >= 0 and 'In' in line and 'mean=' in line:
            match = re.search(r'mean=([-\deE.+]+).*?val=\[([-\deE.+]+)\s+([-\deE.+]+)', line)
            if match:
                try:
                    layers[current_layer]['inputs'].append({
                        'mean': float(match.group(1)),
                        'val0': float(match.group(2)),
                        'val1': float(match.group(3))
                    })
                except ValueError:
                    pass  # Skip invalid values like -nan
    
    return layers

onnx_layers = extract_outputs(onnx_lines)
spinn_layers = extract_outputs(spinn_lines)

print(f"\nONNX有 {len(onnx_layers)} 层, SPINN有 {len(spinn_layers)} 层\n")

for i in range(min(len(onnx_layers), len(spinn_layers))):
    print(f"\n{'=' * 100}")
    print(f"Layer {i}")
    print(f"{'=' * 100}")
    
    if i in onnx_layers and i in spinn_layers:
        o_layer = onnx_layers[i]
        s_layer = spinn_layers[i]
        
        # 对比输入
        if o_layer['inputs'] and s_layer['inputs']:
            print(f"\n输入对比:")
            for j in range(min(len(o_layer['inputs']), len(s_layer['inputs']))):
                o_in = o_layer['inputs'][j]
                s_in = s_layer['inputs'][j]
                mean_diff = abs(o_in['mean'] - s_in['mean'])
                val0_diff = abs(o_in['val0'] - s_in['val0'])
                
                status = "✅" if mean_diff < 0.01 else ("⚠️" if mean_diff < 0.1 else "❌")
                print(f"  In{j}: {status}")
                print(f"    ONNX: mean={o_in['mean']:9.6f}, val=[{o_in['val0']:8.4f}, {o_in['val1']:8.4f}]")
                print(f"    SPINN: mean={s_in['mean']:9.6f}, val=[{s_in['val0']:8.4f}, {s_in['val1']:8.4f}]")
                print(f"    Diff:  mean={mean_diff:9.6f}, val0={val0_diff:8.4f}")
        
        # 对比输出
        if o_layer['outputs'] and s_layer['outputs']:
            print(f"\n输出对比:")
            o_out = o_layer['outputs'][0]
            s_out = s_layer['outputs'][0]
            mean_diff = abs(o_out['mean'] - s_out['mean'])
            val0_diff = abs(o_out['val0'] - s_out['val0'])
            
            status = "✅" if mean_diff < 0.01 else ("⚠️" if mean_diff < 0.1 else "❌")
            print(f"  Out: {status}")
            print(f"    ONNX: mean={o_out['mean']:9.6f}, val=[{o_out['val0']:8.4f}, {o_out['val1']:8.4f}]")
            print(f"    SPINN: mean={s_out['mean']:9.6f}, val=[{s_out['val0']:8.4f}, {s_out['val1']:8.4f}]")
            print(f"    Diff:  mean={mean_diff:9.6f}, val0={val0_diff:8.4f}")
