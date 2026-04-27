# SPINN - 星载嵌入式神经网络推理框架

SPINN (Spaceborne Portable Inference Neural Network) 是面向星载平台的轻量级神经网络推理框架，支持将 ONNX 模型转换为紧凑的二进制格式并高效执行推理。

## 特性

- **紧凑格式 ENNF**：整数 ID 替代字符串引用，元数据开销低至 0.02%~5%
- **静态内存规划**：区间打包算法，FeatureMap 内存节省 92%+
- **图优化**：Conv+BN 融合、Conv+ReLU 融合
- **零依赖运行时**：仅依赖 libc/libm，二进制 522KB
- **213 个 ONNX 算子**：AVX2+FMA SIMD 优化

## 项目结构

```
SPINN/
├── SPINN/                    # 核心代码
│   ├── onnx2ennf.c           # ONNX → ENNF 转换器
│   ├── graph_opt.c           # 图优化（算子融合）
│   ├── ennf_def.h            # ENNF 格式定义
│   └── run_time/             # 运行时引擎
│       ├── spinn_runtime.c   # 加载/规划/执行核心
│       ├── spinn_memory_planner.c  # 静态内存规划
│       ├── spinn_ops.c       # 算子调度
│       └── ops/              # 213 个算子实现
└── README.md
```

## 快速开始

```bash
# 转换 ONNX 模型
./onnx2ennf model.onnx model.ennf

# 运行推理
./spinn_run model.ennf
```

## 已验证模型

- MNIST
- YOLOv10n
- ResNet101

## 许可证

MIT License
