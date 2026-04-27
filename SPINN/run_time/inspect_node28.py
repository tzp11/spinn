#!/usr/bin/env python3
import struct

with open("../complex_test.ennf", "rb") as f:
    header = f.read(64)
    num_nodes = struct.unpack("<I", header[8:12])[0]
    node_table_offset = struct.unpack("<I", header[36:40])[0]
    
    f.seek(node_table_offset)
    
    for i in range(num_nodes):
        base = f.read(8)
        op_type, num_inputs, num_outputs, params_size, flags, reserved = struct.unpack("<HBBHBB", base)
        
        # Skip ids
        f.seek(num_inputs * 2 + num_outputs * 2, 1)
        
        # Read params
        params = f.read(params_size)
        
        if op_type == 235: # ReduceMean (assuming 235, checking spinn_ops for ID)
             # Wait, I need to know OP ID.
             pass

# Let's just dump node 28.
print("Node 28 inspection:")
with open("../complex_test.ennf", "rb") as f:
    header = f.read(64)
    num_nodes = struct.unpack("<I", header[8:12])[0]
    node_table_offset = struct.unpack("<I", header[36:40])[0]
    
    f.seek(node_table_offset)
    for i in range(num_nodes):
        base = f.read(8)
        op_type, num_inputs, num_outputs, params_size, flags, reserved = struct.unpack("<HBBHBB", base)
        
        input_ids = struct.unpack(f"<{num_inputs}H", f.read(num_inputs * 2))
        output_ids = struct.unpack(f"<{num_outputs}H", f.read(num_outputs * 2))
        
        params_val = f.read(params_size)
        
        if i == 28:
            print(f"OpType: {op_type}")
            print(f"Params Size: {params_size}")
            print(f"Inputs: {input_ids}")
            print(f"Outputs: {output_ids}")
            if params_size >= 4: # ReduceParams
                # int32 axes[ENNF_MAX_DIMS]; uint8 num_axes; uint8 keepdims; ...
                # ENNF_MAX_DIMS is 8 usually. 8*4 = 32 bytes for axes.
                # Let's print raw bytes.
                print(f"Raw Params: {params_val.hex()}")
                
                # Manual parse attempt
                # axes: 32 bytes (8 ints)
                # num_axes: 1 byte
                # keepdims: 1 byte
                if len(params_val) >= 34:
                    num_axes = params_val[32]
                    keepdims = params_val[33]
                    print(f"Num Axes: {num_axes}")
                    print(f"Keep Dims: {keepdims}")
                    axes = []
                    for j in range(num_axes):
                        ax = struct.unpack("<i", params_val[j*4:(j+1)*4])[0]
                        axes.append(ax)
                    print(f"Axes: {axes}")
