import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx

class ComplexModel(nn.Module):
    def __init__(self):
        super(ComplexModel, self).__init__()
        
        # 1. CNN 部分
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 2. Residual Block (Shortcut)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        
        # 3. Transformer 部分 (简化版)
        # 假设输入 feature map 是 (B, 16, H, W) -> Flatten -> (B, L, D)
        self.embed_dim = 16
        self.num_heads = 2
        self.head_dim = self.embed_dim // self.num_heads
        
        # Q, K, V Projections
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
        self.ln = nn.LayerNorm(self.embed_dim)
        self.fc = nn.Linear(self.embed_dim, 10) # Output logits

    def forward(self, x):
        # --- CNN Block ---
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # [1, 16, 16, 16]
        
        # --- Shortcut Block ---
        identity = x
        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)
        x = out + identity # <--- OP_Add (Shortcut)
        
        # --- Prepare for Transformer ---
        # [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C] (Sequence)
        b, c, h, w = x.shape
        x = x.view(b, c, h * w) 
        x = x.permute(0, 2, 1) # [B, SeqLen, EmbedDim]
        
        # --- Transformer Block (Self-Attention) ---
        residual = x
        x = self.ln(x)
        
        q = self.q_proj(x) # [B, L, D]
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head: [B, L, Heads, HeadDim] -> [B, Heads, L, HeadDim]
        # 这里会产生 Reshape 和 Transpose 算子
        q = q.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention Score: Q @ K^T
        # [B, Heads, L, HeadDim] @ [B, Heads, HeadDim, L] -> [B, Heads, L, L]
        # 这里会产生 MatMul 算子
        attn = torch.matmul(q, k.transpose(-2, -1))
        
        # Scale (Div)
        attn = attn / (self.head_dim ** 0.5)
        
        # Softmax
        attn = F.softmax(attn, dim=-1)
        
        # Value aggregation: Attn @ V
        out = torch.matmul(attn, v) # [B, Heads, L, HeadDim]
        
        # Merge Heads
        out = out.transpose(1, 2).contiguous().view(b, -1, self.embed_dim)
        
        # Add (Shortcut 2)
        x = residual + out
        
        # Classifier
        # Global Average Pool equivalent: mean over sequence
        x = x.mean(dim=1) # [B, D] -> OP_ReduceMean
        x = self.fc(x)
        
        return x

# Export
model = ComplexModel()
model.eval()

dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(model, dummy_input, "complex_test.onnx", 
                  opset_version=12,
                  input_names=['input'], output_names=['output'])

print("Generated complex_test.onnx")
