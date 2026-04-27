#!/usr/bin/env python3
"""逐层对比 ONNX 和 SPINN 的执行结果"""

import re
import sys

def parse_tensor_stats(line):
    """解析统计行，返回 dict"""
    # 格式: In0 (W): shape=[...] mean=... std=... min=... max=... val=[... ...]
    match = re.search(r'mean=([-\d.]+)\s+std=([-\d.]+)\s+min=([-\d.]+)\s+max=([-\d.]+)', line)
    if match:
        return {
            'mean': float(match.group(1)),
            'std': float(match.group(2)),
            'min': float(match.group(3)),
            'max': float(match.group(4))
        }
    return None

def compare_files(onnx_file, spinn_file):
    """对比两个trace文件"""
    with open(onnx_file) as f:
        onnx_lines = f.readlines()
    with open(spinn_file) as f:
        spinn_lines = f.readlines()
    
    print("=" * 80)
    print("逐层对比分析 (ONNX vs SPINN)")
    print("=" * 80)
    
    onnx_node = -1
    spinn_node = -1
    onnx_stats = {}
    spinn_stats = {}
    
    max_diff_layer = -1
    max_diff_value = 0
    
    for i, (ol, sl) in enumerate(zip(onnx_lines, spinn_lines)):
        # 检测节点开始
        onnx_match = re.search(r'Node (\d+) \(Op (\w+)\):', ol)
        spinn_match = re.search(r'Node (\d+) \(Op (\d+)\):', sl)
        
        if onnx_match and spinn_match:
            onnx_node = int(onnx_match.group(1))
            spinn_node = int(spinn_match.group(1))
            
            if onnx_node != spinn_node:
                print(f"⚠️  警告: 节点序号不匹配! ONNX={onnx_node} SPINN={spinn_node}")
            
            print(f"\n{'─' * 80}")
            print(f"Layer {onnx_node}: {onnx_match.group(2)}")
            onnx_stats = {}
            spinn_stats = {}
            continue
        
        # 解析统计信息
        if 'mean=' in ol:
            parts = ol.strip().split(':')
            if len(parts) >= 1:
                key = parts[0].strip()
                stats = parse_tensor_stats(ol)
                if stats:
                    onnx_stats[key] = stats
        
        if 'mean=' in sl:
            parts = sl.strip().split(':')
            if len(parts) >= 1:
                key = parts[0].strip()
                # SPINN 格式: "In T18 (A):" 需要转换为 "In0"
                key_match = re.search(r'(In|Out)(\d*)\s+T\d+', key)
                if key_match:
                    key = key_match.group(1) + (key_match.group(2) if key_match.group(2) else '0')
                    # 去掉序号前的0
                    key = re.sub(r'([A-Za-z]+)0+(\d)', r'\1\2', key)
                    if key.endswith('0') and len(key) > 3:
                        key = key[:-1]
                    if key == 'In' or key == 'Out':
                        key = key + '0'
                    
                stats = parse_tensor_stats(sl)
                if stats:
                    spinn_stats[key] = stats
        
        # 如果收集到了完整的一层数据，进行对比
        if onnx_stats and spinn_stats and 'Out0' in onnx_stats and 'Out0' in spinn_stats:
            # 对比输出
            o_out = onnx_stats['Out0']
            s_out = spinn_stats['Out0']
            
            mean_diff = abs(o_out['mean'] - s_out['mean'])
            std_diff = abs(o_out['std'] - s_out['std'])
            
            if mean_diff > max_diff_value:
                max_diff_value = mean_diff
                max_diff_layer = onnx_node
            
            status = "✅" if mean_diff < 0.01 else ("⚠️" if mean_diff < 0.1 else "❌")
            
            print(f"{status} Output Mean: ONNX={o_out['mean']:9.6f} SPINN={s_out['mean']:9.6f} Diff={mean_diff:.6f}")
            print(f"   Output Std:  ONNX={o_out['std']:9.6f} SPINN={s_out['std']:9.6f} Diff={std_diff:.6f}")
            
            # 检查权重
            for key in onnx_stats:
                if '(W)' in str(key) or key.startswith('In'):
                    if key in spinn_stats:
                        o_w = onnx_stats[key]
                        s_w = spinn_stats[key]
                        w_mean_diff = abs(o_w['mean'] - s_w['mean'])
                        if w_mean_diff > 0.0001:
                            print(f"   ⚠️  Weight {key} Mean Diff: {w_mean_diff:.6f}")
    
    print("\n" + "=" * 80)
    print(f"最大差异出现在 Layer {max_diff_layer}, Mean Diff = {max_diff_value:.6f}")
    print("=" * 80)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: compare_layers.py onnx_trace.txt spinn_trace.txt")
        sys.exit(1)
    
    compare_files(sys.argv[1], sys.argv[2])
