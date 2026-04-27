/*
 * ennf_op_types.h - ENNF 算子类型枚举
 * 
 * 包含 ONNX 所有标准算子 (200+)
 * 按类别组织，便于维护
 */

#ifndef __ENNF_OP_TYPES_H__
#define __ENNF_OP_TYPES_H__

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * 算子类型枚举
 * 按功能分类，每类预留空间便于扩展
 * ============================================================ */
typedef enum {
    /* ===== 未知/无效 (0) ===== */
    OP_UNKNOWN = 0,
    
    /* ===== 基础数学运算 (1-49) ===== */
    OP_Abs = 1,
    OP_Acos,
    OP_Acosh,
    OP_Add,
    OP_And,
    OP_ArgMax,
    OP_ArgMin,
    OP_Asin,
    OP_Asinh,
    OP_Atan,
    OP_Atanh,
    OP_BitShift,
    OP_BitwiseAnd,
    OP_BitwiseNot,
    OP_BitwiseOr,
    OP_BitwiseXor,
    OP_Ceil,
    OP_Clip,
    OP_Cos,
    OP_Cosh,
    OP_CumSum,
    OP_Det,
    OP_Div,
    OP_Equal,
    OP_Erf,
    OP_Exp,
    OP_Floor,
    OP_Greater,
    OP_GreaterOrEqual,
    OP_IsInf,
    OP_IsNaN,
    OP_Less,
    OP_LessOrEqual,
    OP_Log,
    OP_Max,
    OP_Mean,
    OP_Min,
    OP_Mod,
    OP_Mul,
    OP_Neg,
    OP_Not,
    OP_Or,
    OP_Pow,
    OP_Reciprocal,
    OP_Round,
    OP_Sign,
    OP_Sin,
    OP_Sinh,
    OP_Sqrt,
    OP_Sub,
    OP_Sum,
    OP_Tan,
    OP_Xor,
    
    /* ===== 卷积类 (60-79) ===== */
    OP_Conv = 60,
    OP_ConvInteger,
    OP_ConvTranspose,
    OP_QLinearConv,
    OP_DeformConv,
    OP_DepthToSpace,
    OP_SpaceToDepth,
    OP_Im2Col,
    OP_Col2Im,
    
    /* ===== 池化类 (80-99) ===== */
    OP_AveragePool = 80,
    OP_GlobalAveragePool,
    OP_GlobalLpPool,
    OP_GlobalMaxPool,
    OP_LpPool,
    OP_MaxPool,
    OP_MaxRoiPool,
    OP_MaxUnpool,
    OP_RoiAlign,
    OP_AdaptiveAvgPool2D,
    
    /* ===== 归一化类 (100-119) ===== */
    OP_BatchNormalization = 100,
    OP_GroupNormalization,
    OP_InstanceNormalization,
    OP_LayerNormalization,
    OP_LRN,
    OP_LpNormalization,
    OP_MeanVarianceNormalization,
    OP_RMSNormalization,
    
    /* ===== 激活函数 (120-159) ===== */
    OP_Celu = 120,
    OP_Elu,
    OP_Gelu,
    OP_HardSigmoid,
    OP_HardSwish,
    OP_Hardmax,
    OP_LeakyRelu,
    OP_LogSoftmax,
    OP_Mish,
    OP_PRelu,
    OP_Relu,
    OP_Selu,
    OP_Shrink,
    OP_Sigmoid,
    OP_Softmax,
    OP_Softplus,
    OP_Softsign,
    OP_Swish,
    OP_Tanh,
    OP_ThresholdedRelu,
    
    /* ===== 矩阵运算 (160-179) ===== */
    OP_Gemm = 160,
    OP_MatMul,
    OP_MatMulInteger,
    OP_QLinearMatMul,
    OP_Einsum,
    OP_QGemm,
    
    /* ===== Tensor 操作 (180-229) ===== */
    OP_Cast = 180,
    OP_CastLike,
    OP_Compress,
    OP_Concat,
    OP_ConcatFromSequence,
    OP_Constant,
    OP_ConstantOfShape,
    OP_Expand,
    OP_EyeLike,
    OP_Flatten,
    OP_Gather,
    OP_GatherElements,
    OP_GatherND,
    OP_Identity,
    OP_NonZero,
    OP_OneHot,
    OP_Pad,
    OP_Reshape,
    OP_Resize,
    OP_ReverseSequence,
    OP_Scatter,
    OP_ScatterElements,
    OP_ScatterND,
    OP_Shape,
    OP_Size,
    OP_Slice,
    OP_Split,
    OP_SplitToSequence,
    OP_Squeeze,
    OP_Tile,
    OP_TopK,
    OP_Transpose,
    OP_Trilu,
    OP_Unique,
    OP_Unsqueeze,
    OP_Upsample,
    OP_Where,
    OP_CenterCropPad,
    OP_GridSample,
    OP_RegexFullMatch,
    OP_StringConcat,
    OP_StringSplit,
    OP_ImageDecoder,
    
    /* ===== Reduce 类 (230-259) ===== */
    OP_ReduceL1 = 230,
    OP_ReduceL2,
    OP_ReduceLogSum,
    OP_ReduceLogSumExp,
    OP_ReduceMax,
    OP_ReduceMean,
    OP_ReduceMin,
    OP_ReduceProd,
    OP_ReduceSum,
    OP_ReduceSumSquare,
    
    /* ===== RNN 类 (260-279) ===== */
    OP_GRU = 260,
    OP_LSTM,
    OP_RNN,
    
    /* ===== Sequence/Control Flow (280-309) ===== */
    OP_If = 280,
    OP_Loop,
    OP_Scan,
    OP_SequenceAt,
    OP_SequenceConstruct,
    OP_SequenceEmpty,
    OP_SequenceErase,
    OP_SequenceInsert,
    OP_SequenceLength,
    OP_SequenceMap,
    OP_Optional,
    OP_OptionalGetElement,
    OP_OptionalHasElement,
    
    /* ===== 量化类 (310-329) ===== */
    OP_DequantizeLinear = 310,
    OP_DynamicQuantizeLinear,
    OP_QuantizeLinear,
    OP_QLinearAdd,
    OP_QLinearMul,
    OP_QLinearSigmoid,
    OP_QLinearSoftmax,
    OP_QLinearLeakyRelu,
    OP_QLinearConcat,
    OP_QLinearGlobalAveragePool,
    OP_QLinearAveragePool,
    
    /* ===== Transformer/Attention (330-369) ===== */
    OP_Attention = 330,
    OP_RotaryEmbedding,
    OP_MultiHeadAttention,
    OP_GroupQueryAttention,
    OP_PagedAttention,
    OP_SkipLayerNormalization,
    OP_SkipSimplifiedLayerNormalization,
    OP_EmbedLayerNormalization,
    OP_BiasGelu,
    OP_FastGelu,
    OP_FusedMatMul,
    OP_NhwcConv,
    OP_QuickGelu,
    OP_RelativePositionBias,
    OP_PackedAttention,
    OP_PackedMultiHeadAttention,
    OP_SimplifiedLayerNormalization,
    OP_MatMulNBits,
    OP_BiasSplitGelu,
    OP_BiasAdd,
    OP_Gelu_Bias,
    
    /* ===== 随机/采样 (370-389) ===== */
    OP_Bernoulli = 370,
    OP_Multinomial,
    OP_RandomNormal,
    OP_RandomNormalLike,
    OP_RandomUniform,
    OP_RandomUniformLike,
    OP_Range,
    OP_Dropout,
    
    /* ===== 损失函数 (390-409) ===== */
    OP_NegativeLogLikelihoodLoss = 390,
    OP_SoftmaxCrossEntropyLoss,
    
    /* ===== 图像处理 (410-429) ===== */
    OP_NonMaxSuppression = 410,
    OP_AffineGrid,
    
    /* ===== 窗口函数 (430-439) ===== */
    OP_BlackmanWindow = 430,
    OP_HammingWindow,
    OP_HannWindow,
    OP_MelWeightMatrix,
    OP_STFT,
    OP_DFT,
    
    /* ===== 字符串/文本 (440-449) ===== */
    OP_StringNormalizer = 440,
    OP_TfIdfVectorizer,
    
    /* ===== 其他 (450+) ===== */
    OP_TensorScatter = 450,
    
    /* 最大值标记 */
    OP_MAX_OPTYPE
} ENNF_OpType;

/* ============================================================
 * 算子名称查找表
 * ============================================================ */

/* 算子名称到枚举的映射结构 */
typedef struct {
    const char *name;
    ENNF_OpType type;
} ENNF_OpTypeEntry;

/* 获取算子枚举值 (根据ONNX op_type 字符串) */
ENNF_OpType ennf_get_op_type(const char *name);

/* 获取算子名称 (根据枚举值) */
const char* ennf_get_op_name(ENNF_OpType type);

#ifdef __cplusplus
}
#endif

#endif /* __ENNF_OP_TYPES_H__ */
