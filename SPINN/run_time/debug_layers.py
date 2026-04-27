import onnx
import onnxruntime
import numpy as np
import sys
import onnx.numpy_helper

def print_stats(name, data):
    if data is None: return
    data = data.flatten()
    if len(data) == 0: return
    mean = np.mean(data)
    std = np.std(data)
    min_v = np.min(data)
    max_v = np.max(data)
    print(f"    {name}: shape={data.shape} mean={mean:.6f} std={std:.6f} min={min_v:.6f} max={max_v:.6f} val=[{data[0]:.4f} {data[1] if len(data)>1 else 0:.4f}]")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: debug_layers.py <onnx_file>")
        sys.exit(1)
        
    onnx_file = sys.argv[1]
    model = onnx.load(onnx_file)
    
    # 提取所有中间 Tensor
    intermediate_tensor_names = []
    seen = set()
    for node in model.graph.node:
        for output in node.output:
            if output not in seen:
                intermediate_tensor_names.append(output)
                seen.add(output)
    
    # 添加到 Graph Output
    existing_outs = {o.name for o in model.graph.output}
    for name in intermediate_tensor_names:
        if name not in existing_outs:
            vi = onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
            model.graph.output.append(vi)

    tmp_model = "debug_tmp.onnx"
    onnx.save(model, tmp_model)
    
    ort_session = onnxruntime.InferenceSession(tmp_model)
    
    #构造输入
    input_name = ort_session.get_inputs()[0].name
    input_shape = ort_session.get_inputs()[0].shape
    shape = [d if isinstance(d, int) else 1 for d in input_shape]
    total_elems = 1
    for d in shape: total_elems *= d
    
    # 模拟 main.c 的输入生成逻辑: i / 1000.0
    data = np.array([i / 1000.0 for i in range(total_elems)], dtype=np.float32).reshape(shape)
    
    # Run
    sess_out_names = [o.name for o in ort_session.get_outputs()]
    outputs = ort_session.run(sess_out_names, {input_name: data})
    out_map = {name: val for name, val in zip(sess_out_names, outputs)}
    
    print("ONNX Layer Stats:")
    weights = {t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer}

    for i, node in enumerate(model.graph.node):
        print(f"  Node {i} (Op {node.op_type}):")
        # Inputs
        for j, inp in enumerate(node.input):
            if inp in out_map:
                print_stats(f"In{j}", out_map[inp])
            elif inp in weights:
                print_stats(f"In{j} (W)", weights[inp])
            elif inp == input_name:
                print_stats(f"In{j}", data)
            else:
                print(f"    In{j}: (Unknown)")
        # Outputs
        for j, out in enumerate(node.output):
             if out in out_map:
                 print_stats(f"Out{j}", out_map[out])
