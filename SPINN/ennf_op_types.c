/*
 * ennf_op_types.c - 算子类型查找表实现
 */

#include <string.h>
#include "ennf_op_types.h"

/* 算子名称到枚举值的映射表 */
static const ENNF_OpTypeEntry op_type_table[] = {
    /* 基础数学运算 */
    {"Abs", OP_Abs},
    {"Acos", OP_Acos},
    {"Acosh", OP_Acosh},
    {"Add", OP_Add},
    {"And", OP_And},
    {"ArgMax", OP_ArgMax},
    {"ArgMin", OP_ArgMin},
    {"Asin", OP_Asin},
    {"Asinh", OP_Asinh},
    {"Atan", OP_Atan},
    {"Atanh", OP_Atanh},
    {"BitShift", OP_BitShift},
    {"BitwiseAnd", OP_BitwiseAnd},
    {"BitwiseNot", OP_BitwiseNot},
    {"BitwiseOr", OP_BitwiseOr},
    {"BitwiseXor", OP_BitwiseXor},
    {"Ceil", OP_Ceil},
    {"Clip", OP_Clip},
    {"Cos", OP_Cos},
    {"Cosh", OP_Cosh},
    {"CumSum", OP_CumSum},
    {"Det", OP_Det},
    {"Div", OP_Div},
    {"Equal", OP_Equal},
    {"Erf", OP_Erf},
    {"Exp", OP_Exp},
    {"Floor", OP_Floor},
    {"Greater", OP_Greater},
    {"GreaterOrEqual", OP_GreaterOrEqual},
    {"IsInf", OP_IsInf},
    {"IsNaN", OP_IsNaN},
    {"Less", OP_Less},
    {"LessOrEqual", OP_LessOrEqual},
    {"Log", OP_Log},
    {"Max", OP_Max},
    {"Mean", OP_Mean},
    {"Min", OP_Min},
    {"Mod", OP_Mod},
    {"Mul", OP_Mul},
    {"Neg", OP_Neg},
    {"Not", OP_Not},
    {"Or", OP_Or},
    {"Pow", OP_Pow},
    {"Reciprocal", OP_Reciprocal},
    {"Round", OP_Round},
    {"Sign", OP_Sign},
    {"Sin", OP_Sin},
    {"Sinh", OP_Sinh},
    {"Sqrt", OP_Sqrt},
    {"Sub", OP_Sub},
    {"Sum", OP_Sum},
    {"Tan", OP_Tan},
    {"Xor", OP_Xor},
    
    /* 卷积类 */
    {"Conv", OP_Conv},
    {"ConvInteger", OP_ConvInteger},
    {"ConvTranspose", OP_ConvTranspose},
    {"QLinearConv", OP_QLinearConv},
    {"DeformConv", OP_DeformConv},
    {"DepthToSpace", OP_DepthToSpace},
    {"SpaceToDepth", OP_SpaceToDepth},
    {"Im2Col", OP_Im2Col},
    {"Col2Im", OP_Col2Im},
    
    /* 池化类 */
    {"AveragePool", OP_AveragePool},
    {"GlobalAveragePool", OP_GlobalAveragePool},
    {"GlobalLpPool", OP_GlobalLpPool},
    {"GlobalMaxPool", OP_GlobalMaxPool},
    {"LpPool", OP_LpPool},
    {"MaxPool", OP_MaxPool},
    {"MaxRoiPool", OP_MaxRoiPool},
    {"MaxUnpool", OP_MaxUnpool},
    {"RoiAlign", OP_RoiAlign},
    
    /* 归一化类 */
    {"BatchNormalization", OP_BatchNormalization},
    {"GroupNormalization", OP_GroupNormalization},
    {"InstanceNormalization", OP_InstanceNormalization},
    {"LayerNormalization", OP_LayerNormalization},
    {"LRN", OP_LRN},
    {"LpNormalization", OP_LpNormalization},
    {"MeanVarianceNormalization", OP_MeanVarianceNormalization},
    {"RMSNormalization", OP_RMSNormalization},
    
    /* 激活函数 */
    {"Celu", OP_Celu},
    {"Elu", OP_Elu},
    {"Gelu", OP_Gelu},
    {"HardSigmoid", OP_HardSigmoid},
    {"HardSwish", OP_HardSwish},
    {"Hardmax", OP_Hardmax},
    {"LeakyRelu", OP_LeakyRelu},
    {"LogSoftmax", OP_LogSoftmax},
    {"Mish", OP_Mish},
    {"PRelu", OP_PRelu},
    {"Relu", OP_Relu},
    {"Selu", OP_Selu},
    {"Shrink", OP_Shrink},
    {"Sigmoid", OP_Sigmoid},
    {"Softmax", OP_Softmax},
    {"Softplus", OP_Softplus},
    {"Softsign", OP_Softsign},
    {"Swish", OP_Swish},
    {"Tanh", OP_Tanh},
    {"ThresholdedRelu", OP_ThresholdedRelu},
    
    /* 矩阵运算 */
    {"Gemm", OP_Gemm},
    {"MatMul", OP_MatMul},
    {"MatMulInteger", OP_MatMulInteger},
    {"QLinearMatMul", OP_QLinearMatMul},
    {"Einsum", OP_Einsum},
    
    /* Tensor 操作 */
    {"Cast", OP_Cast},
    {"CastLike", OP_CastLike},
    {"Compress", OP_Compress},
    {"Concat", OP_Concat},
    {"ConcatFromSequence", OP_ConcatFromSequence},
    {"Constant", OP_Constant},
    {"ConstantOfShape", OP_ConstantOfShape},
    {"Expand", OP_Expand},
    {"EyeLike", OP_EyeLike},
    {"Flatten", OP_Flatten},
    {"Gather", OP_Gather},
    {"GatherElements", OP_GatherElements},
    {"GatherND", OP_GatherND},
    {"Identity", OP_Identity},
    {"NonZero", OP_NonZero},
    {"OneHot", OP_OneHot},
    {"Pad", OP_Pad},
    {"Reshape", OP_Reshape},
    {"Resize", OP_Resize},
    {"ReverseSequence", OP_ReverseSequence},
    {"Scatter", OP_Scatter},
    {"ScatterElements", OP_ScatterElements},
    {"ScatterND", OP_ScatterND},
    {"Shape", OP_Shape},
    {"Size", OP_Size},
    {"Slice", OP_Slice},
    {"Split", OP_Split},
    {"SplitToSequence", OP_SplitToSequence},
    {"Squeeze", OP_Squeeze},
    {"Tile", OP_Tile},
    {"TopK", OP_TopK},
    {"Transpose", OP_Transpose},
    {"Trilu", OP_Trilu},
    {"Unique", OP_Unique},
    {"Unsqueeze", OP_Unsqueeze},
    {"Upsample", OP_Upsample},
    {"Where", OP_Where},
    {"CenterCropPad", OP_CenterCropPad},
    {"GridSample", OP_GridSample},
    
    /* Reduce 类 */
    {"ReduceL1", OP_ReduceL1},
    {"ReduceL2", OP_ReduceL2},
    {"ReduceLogSum", OP_ReduceLogSum},
    {"ReduceLogSumExp", OP_ReduceLogSumExp},
    {"ReduceMax", OP_ReduceMax},
    {"ReduceMean", OP_ReduceMean},
    {"ReduceMin", OP_ReduceMin},
    {"ReduceProd", OP_ReduceProd},
    {"ReduceSum", OP_ReduceSum},
    {"ReduceSumSquare", OP_ReduceSumSquare},
    
    /* RNN 类 */
    {"GRU", OP_GRU},
    {"LSTM", OP_LSTM},
    {"RNN", OP_RNN},
    
    /* Sequence/Control Flow */
    {"If", OP_If},
    {"Loop", OP_Loop},
    {"Scan", OP_Scan},
    {"SequenceAt", OP_SequenceAt},
    {"SequenceConstruct", OP_SequenceConstruct},
    {"SequenceEmpty", OP_SequenceEmpty},
    {"SequenceErase", OP_SequenceErase},
    {"SequenceInsert", OP_SequenceInsert},
    {"SequenceLength", OP_SequenceLength},
    {"SequenceMap", OP_SequenceMap},
    {"Optional", OP_Optional},
    {"OptionalGetElement", OP_OptionalGetElement},
    {"OptionalHasElement", OP_OptionalHasElement},
    
    /* 量化类 */
    {"DequantizeLinear", OP_DequantizeLinear},
    {"DynamicQuantizeLinear", OP_DynamicQuantizeLinear},
    {"QuantizeLinear", OP_QuantizeLinear},
    
    /* Transformer/Attention */
    {"Attention", OP_Attention},
    {"RotaryEmbedding", OP_RotaryEmbedding},
    {"MultiHeadAttention", OP_MultiHeadAttention},
    {"GroupQueryAttention", OP_GroupQueryAttention},
    {"SkipLayerNormalization", OP_SkipLayerNormalization},
    {"EmbedLayerNormalization", OP_EmbedLayerNormalization},
    {"BiasGelu", OP_BiasGelu},
    {"FastGelu", OP_FastGelu},
    {"FusedMatMul", OP_FusedMatMul},
    
    /* 随机/采样 */
    {"Bernoulli", OP_Bernoulli},
    {"Multinomial", OP_Multinomial},
    {"RandomNormal", OP_RandomNormal},
    {"RandomNormalLike", OP_RandomNormalLike},
    {"RandomUniform", OP_RandomUniform},
    {"RandomUniformLike", OP_RandomUniformLike},
    {"Range", OP_Range},
    {"Dropout", OP_Dropout},
    
    /* 损失函数 */
    {"NegativeLogLikelihoodLoss", OP_NegativeLogLikelihoodLoss},
    {"SoftmaxCrossEntropyLoss", OP_SoftmaxCrossEntropyLoss},
    
    /* 图像处理 */
    {"NonMaxSuppression", OP_NonMaxSuppression},
    {"AffineGrid", OP_AffineGrid},
    
    /* 窗口函数 */
    {"BlackmanWindow", OP_BlackmanWindow},
    {"HammingWindow", OP_HammingWindow},
    {"HannWindow", OP_HannWindow},
    {"MelWeightMatrix", OP_MelWeightMatrix},
    {"STFT", OP_STFT},
    {"DFT", OP_DFT},
    
    /* 字符串/文本 */
    {"StringNormalizer", OP_StringNormalizer},
    {"TfIdfVectorizer", OP_TfIdfVectorizer},
    
    /* 其他 */
    {"TensorScatter", OP_TensorScatter},
    
    /* 结束标记 */
    {NULL, OP_UNKNOWN}
};

/* 算子枚举到名称的反向映射 (懒加载，可选) */
static const char* op_names[OP_MAX_OPTYPE] = {NULL};
static int op_names_initialized = 0;

static void init_op_names(void) {
    if (op_names_initialized) return;
    for (int i = 0; op_type_table[i].name != NULL; i++) {
        if (op_type_table[i].type < OP_MAX_OPTYPE) {
            op_names[op_type_table[i].type] = op_type_table[i].name;
        }
    }
    op_names_initialized = 1;
}

ENNF_OpType ennf_get_op_type(const char *name) {
    if (!name) return OP_UNKNOWN;
    
    for (int i = 0; op_type_table[i].name != NULL; i++) {
        if (strcmp(op_type_table[i].name, name) == 0) {
            return op_type_table[i].type;
        }
    }
    return OP_UNKNOWN;
}

const char* ennf_get_op_name(ENNF_OpType type) {
    init_op_names();
    if (type >= 0 && type < OP_MAX_OPTYPE && op_names[type]) {
        return op_names[type];
    }
    return "Unknown";
}
