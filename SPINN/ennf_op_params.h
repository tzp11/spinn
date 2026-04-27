/*
 * ennf_op_params.h - ENNF 算子参数结构体定义
 * 
 * 每个算子的参数结构体，用于序列化/反序列化
 * 所有结构体按 4 字节对齐
 */

#ifndef __ENNF_OP_PARAMS_H__
#define __ENNF_OP_PARAMS_H__

#include <stdint.h>
#include "ennf_def.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * 卷积类参数
 * ============================================================ */

/* Conv, ConvTranspose, QLinearConv */
typedef struct {
    uint16_t kernel_shape[4];   /* 卷积核尺寸 (最多4D) */
    uint16_t strides[4];        /* 步长 */
    uint16_t pads[8];           /* padding (每维前后各一个) */
    uint16_t dilations[4];      /* 膨胀率 */
    uint16_t group;             /* 分组数 */
    uint8_t  auto_pad;          /* ENNF_AutoPad 枚举 */
    uint8_t  reserved;
} ENNF_ConvParams;

/* ConvTranspose 额外参数 */
typedef struct {
    ENNF_ConvParams base;
    uint16_t output_padding[4]; /* 输出 padding */
    uint16_t output_shape[4];   /* 目标输出尺寸 (可选) */
} ENNF_ConvTransposeParams;

/* ============================================================
 * 池化类参数
 * ============================================================ */

/* MaxPool, AveragePool, LpPool */
typedef struct {
    uint16_t kernel_shape[4];   /* 池化窗口尺寸 */
    uint16_t strides[4];        /* 步长 */
    uint16_t pads[8];           /* padding */
    uint16_t dilations[4];      /* 膨胀率 (MaxPool) */
    uint8_t  auto_pad;          /* ENNF_AutoPad */
    uint8_t  ceil_mode;         /* 是否向上取整 */
    uint8_t  count_include_pad; /* AveragePool: 是否包含 padding */
    uint8_t  storage_order;     /* MaxPool: indices 存储顺序 */
    int32_t  p;                 /* LpPool: Lp 范数的 p 值 */
} ENNF_PoolParams;

/* ============================================================
 * 归一化类参数
 * ============================================================ */

/* BatchNormalization */
typedef struct {
    float    epsilon;
    float    momentum;
    uint8_t  training_mode;
    uint8_t  reserved[3];
} ENNF_BatchNormParams;

/* LayerNormalization, GroupNormalization */
typedef struct {
    float    epsilon;
    int32_t  axis;              /* 归一化起始轴 */
    uint8_t  stash_type;        /* 中间结果类型 */
    uint8_t  reserved[3];
    int32_t  num_groups;        /* GroupNorm: 组数 */
} ENNF_LayerNormParams;

/* LRN (Local Response Normalization) */
typedef struct {
    float    alpha;
    float    beta;
    float    bias;
    int32_t  size;
} ENNF_LRNParams;

/* ============================================================
 * 激活函数参数
 * ============================================================ */

/* LeakyRelu */
typedef struct {
    float alpha;
} ENNF_LeakyReluParams;

/* Elu, Selu, Celu */
typedef struct {
    float alpha;
    float gamma;                /* Selu 专用 */
} ENNF_EluParams;

/* Clip */
typedef struct {
    float min_val;
    float max_val;
} ENNF_ClipParams;

/* HardSigmoid, HardSwish: y = max(0, min(1, alpha*x + beta)) */
typedef struct {
    float alpha;
    float beta;
} ENNF_HardSigmoidParams;

/* Softmax, LogSoftmax */
typedef struct {
    int32_t axis;               /* 计算轴 */
} ENNF_SoftmaxParams;

/* Shrink */
typedef struct {
    float bias;
    float lambd;
} ENNF_ShrinkParams;

/* ThresholdedRelu */
typedef struct {
    float alpha;
} ENNF_ThresholdedReluParams;

/* ============================================================
 * 矩阵运算参数
 * ============================================================ */

/* Gemm: Y = alpha * A * B + beta * C */
typedef struct {
    float    alpha;
    float    beta;
    uint8_t  transA;            /* 是否转置 A */
    uint8_t  transB;            /* 是否转置 B */
    uint8_t  reserved[2];
} ENNF_GemmParams;

/* ============================================================
 * Tensor 操作参数
 * ============================================================ */

/* Transpose */
typedef struct {
    uint8_t  perm[ENNF_MAX_DIMS]; /* 维度排列 */
    uint8_t  num_dims;
    uint8_t  reserved[3];
} ENNF_TransposeParams;

/* Flatten */
typedef struct {
    int32_t axis;
} ENNF_FlattenParams;

/* Squeeze, Unsqueeze */
typedef struct {
    int32_t axes[ENNF_MAX_DIMS];
    uint8_t  num_axes;
    uint8_t  reserved[3];
} ENNF_SqueezeParams;

/* Concat */
typedef struct {
    int32_t axis;
} ENNF_ConcatParams;

/* Split */
typedef struct {
    int32_t axis;
    uint32_t num_outputs;
    /* split sizes 通过输入 tensor 传入 */
} ENNF_SplitParams;

/* Slice */
typedef struct {
    /* starts, ends, axes, steps 通过输入 tensors 传入 */
    /* 这里只存静态信息 */
    uint8_t reserved[4];
} ENNF_SliceParams;

/* Pad */
typedef struct {
    uint8_t  mode;              /* 0=constant, 1=reflect, 2=edge, 3=wrap */
    uint8_t  reserved[3];
    float    constant_value;
    /* pads 通过输入 tensor 传入 */
} ENNF_PadParams;

/* Resize */
typedef struct {
    uint8_t  mode;              /* 0=nearest, 1=linear, 2=cubic */
    uint8_t  coord_transform_mode; /* 坐标变换模式 */
    uint8_t  nearest_mode;      /* nearest 采样模式 */
    uint8_t  antialias;
    float    cubic_coeff_a;
    uint8_t  exclude_outside;
    uint8_t  keep_aspect_ratio_policy;
    uint8_t  reserved[2];
    /* scales, sizes, roi 通过输入 tensors 传入 */
} ENNF_ResizeParams;

/* Gather, GatherElements, GatherND */
typedef struct {
    int32_t axis;
    int32_t batch_dims;         /* GatherND */
} ENNF_GatherParams;

/* Scatter, ScatterElements, ScatterND */
typedef struct {
    int32_t axis;
    uint8_t  reduction;         /* 0=none, 1=add, 2=mul, 3=max, 4=min */
    uint8_t  reserved[3];
} ENNF_ScatterParams;

/* TopK */
typedef struct {
    int32_t axis;
    uint8_t  largest;           /* 是否取最大 */
    uint8_t  sorted;            /* 是否排序 */
    uint8_t  reserved[2];
} ENNF_TopKParams;

/* OneHot */
typedef struct {
    int32_t axis;
} ENNF_OneHotParams;

/* Tile */
typedef struct {
    /* repeats 通过输入 tensor 传入 */
    uint8_t reserved[4];
} ENNF_TileParams;

/* Cast */
typedef struct {
    uint8_t  to_type;           /* ENNF_DataType */
    uint8_t  saturate;
    uint8_t  reserved[2];
} ENNF_CastParams;

/* Trilu */
typedef struct {
    uint8_t  upper;             /* 是否上三角 */
    uint8_t  reserved[3];
    /* k 通过输入 tensor 传入 */
} ENNF_TriluParams;

/* ============================================================
 * Reduce 类参数
 * ============================================================ */

typedef struct {
    int32_t axes[ENNF_MAX_DIMS];/* 要 reduce 的轴 */
    uint8_t  num_axes;
    uint8_t  keepdims;          /* 是否保持维度 */
    uint8_t  noop_with_empty_axes;
    uint8_t  reserved;
} ENNF_ReduceParams;

/* ============================================================
 * RNN 类参数
 * ============================================================ */

/* LSTM */
typedef struct {
    int64_t  hidden_size;
    uint8_t  direction;         /* 0=forward, 1=reverse, 2=bidirectional */
    uint8_t  input_forget;      /* 是否合并 input 和 forget gate */
    uint8_t  layout;            /* 0=seq_batch_feature, 1=batch_seq_feature */
    uint8_t  reserved;
    float    clip;              /* 梯度裁剪阈值 */
    /* activation functions 使用默认 */
} ENNF_LSTMParams;

/* GRU */
typedef struct {
    int64_t  hidden_size;
    uint8_t  direction;
    uint8_t  linear_before_reset;
    uint8_t  layout;
    uint8_t  reserved;
    float    clip;
} ENNF_GRUParams;

/* RNN */
typedef struct {
    int64_t  hidden_size;
    uint8_t  direction;
    uint8_t  layout;
    uint8_t  reserved[2];
    float    clip;
    /* activation function name 需要额外处理 */
} ENNF_RNNParams;

/* ============================================================
 * Transformer/Attention 参数
 * ============================================================ */

/* Attention */
typedef struct {
    int64_t  num_heads;
    uint8_t  unidirectional;
    uint8_t  do_rotary;
    uint8_t  rotary_interleaved;
    uint8_t  mask_filter_value_type;
    float    mask_filter_value;
    float    scale;
} ENNF_AttentionParams;

/* LayerNormalization (已在归一化类中定义) */

/* ============================================================
 * 量化类参数
 * ============================================================ */

/* QuantizeLinear, DequantizeLinear */
typedef struct {
    int32_t axis;
    uint8_t  block_size;
    uint8_t  saturate;
    uint8_t  reserved[2];
} ENNF_QuantizeParams;

/* ============================================================
 * 其他算子参数
 * ============================================================ */

/* DepthToSpace, SpaceToDepth */
typedef struct {
    int32_t  blocksize;
    uint8_t  mode;              /* 0=DCR, 1=CRD */
    uint8_t  reserved[3];
} ENNF_SpaceParams;

/* Dropout */
typedef struct {
    float    ratio;
    uint8_t  training_mode;
    uint8_t  reserved[3];
    int64_t  seed;
} ENNF_DropoutParams;

/* RandomNormal, RandomUniform */
typedef struct {
    float    mean;
    float    scale;
    int64_t  seed;
    uint8_t  dtype;             /* 输出数据类型 */
    uint8_t  reserved[3];
    int32_t  shape[ENNF_MAX_DIMS];
    uint8_t  num_dims;
    uint8_t  reserved2[3];
} ENNF_RandomParams;

/* Range */
typedef struct {
    /* start, limit, delta 通过输入 tensors 传入 */
    uint8_t reserved[4];
} ENNF_RangeParams;

/* NonMaxSuppression */
typedef struct {
    int32_t  center_point_box;
} ENNF_NMSParams;

/* Einsum */
typedef struct {
    /* equation 字符串需要特殊处理，这里存偏移量 */
    uint32_t equation_offset;   /* 在字符串表中的偏移 */
    uint16_t equation_length;
    uint8_t  reserved[2];
} ENNF_EinsumParams;

/* Shape */
typedef struct {
    int32_t start;
    int32_t end;
} ENNF_ShapeParams;

/* ConstantOfShape */
typedef struct {
    float    value;             /* 填充值 (简化，实际可能是 tensor) */
    uint8_t  dtype;
    uint8_t  reserved[3];
} ENNF_ConstantOfShapeParams;

/* ============================================================
 * 无参数算子列表
 * ============================================================ */
/*
 * 以下算子不需要参数结构体 (params_size = 0):
 * - Abs, Acos, Acosh, Asin, Asinh, Atan, Atanh
 * - BitwiseAnd, BitwiseNot, BitwiseOr, BitwiseXor
 * - Ceil, Cos, Cosh, Det, Erf, Exp, Floor
 * - Identity, IsInf, IsNaN, Log, Neg, Not
 * - Reciprocal, Relu, Round, Sigmoid, Sign, Sin, Sinh
 * - Sqrt, Tan, Tanh
 * - Add, Sub, Mul, Div, Pow, Mod (二元运算)
 * - And, Or, Xor, Equal, Greater, Less, etc. (比较运算)
 * - GlobalMaxPool, GlobalAveragePool (无参数)
 * - Shape, Size (无参数版本)
 * - Reshape (shape 通过输入传入)
 */

#ifdef __cplusplus
}
#endif

#endif /* __ENNF_OP_PARAMS_H__ */
