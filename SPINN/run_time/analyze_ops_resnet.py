
import torch
import torchvision
import onnx
import warnings

warnings.filterwarnings("ignore")

def analyze_onnx_ops(onnx_path, model_name):
    print(f"\n{'='*20} Analyzing {model_name} {'='*20}")
    try:
        model = onnx.load(onnx_path)
        ops = set()
        for node in model.graph.node:
            ops.add(node.op_type)
        print(f"Total Unique Ops: {len(ops)}")
        print(f"Ops: {sorted(list(ops))}")
        return ops
    except Exception as e:
        print(f"Failed to analyze {onnx_path}: {e}")
        return set()

def export_and_analyze_resnet101():
    print("\n[Exporting ResNet101...]")
    try:
        model = torchvision.models.resnet101(weights=None)
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        onnx_path = "resnet101.onnx"
        torch.onnx.export(model, dummy_input, onnx_path)
        return analyze_onnx_ops(onnx_path, "ResNet101")
    except Exception as e:
        print(f"Error exporting ResNet101: {e}")
        return set()

if __name__ == "__main__":
    current_ops = {
        'Relu', 'Add', 'MatMul', 'Softmax', 'Reshape', 'Transpose', 
        'Conv', 'MaxPool', 'Neg', 'Div', 'InstanceNormalization', 
        'Gemm', 'ReduceMean', 'Equal', 'LayerNormalization'
    }
    
    # 1. ResNet101
    resnet_ops = export_and_analyze_resnet101()
    if resnet_ops:
        missing = resnet_ops - current_ops
        print(f"MISSING for ResNet101 ({len(missing)}): {sorted(list(missing))}")
