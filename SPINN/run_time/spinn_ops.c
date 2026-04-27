/*
 * spinn_ops.c - 算子库实现
 * 
 * 包含调度器和算子注册
 */

#include "spinn_ops.h"
#include "../ennf_op_types.h"
#include <stdio.h>

// 引入拆分后的算子内核
#include "ops/kernels.h"

/* ============================================================
 * 算子注册表
 * ============================================================ */
static SpinnOpKernel g_op_registry[512] = {0};
static int g_registry_initialized = 0;

static void init_registry(void) {
    if (g_registry_initialized) return;
    
    // ========== 基础数学运算 ==========
    g_op_registry[OP_Abs]      = op_abs;
    g_op_registry[OP_Add]      = op_add;
    g_op_registry[OP_And]      = op_and;
    g_op_registry[OP_ArgMax]   = op_argmax;
    g_op_registry[OP_ArgMin]   = op_argmin;
    g_op_registry[OP_Acos]     = op_acos;
    g_op_registry[OP_Acosh]    = op_acosh;
    g_op_registry[OP_AdaptiveAvgPool2D] = op_adaptive_avg_pool; // New
    g_op_registry[OP_Asin]     = op_asin;
    g_op_registry[OP_Asinh]    = op_asinh;
    g_op_registry[OP_Atan]     = op_atan;
    g_op_registry[OP_Atanh]    = op_atanh;
    g_op_registry[OP_BitShift]   = op_bitshift; // New
    g_op_registry[OP_BitwiseAnd] = op_bitwise_and;
    g_op_registry[OP_BitwiseNot] = op_bitwise_not;
    g_op_registry[OP_BitwiseOr]  = op_bitwise_or;
    g_op_registry[OP_BitwiseXor] = op_bitwise_xor;
    g_op_registry[OP_Ceil]     = op_ceil;
    g_op_registry[OP_Clip]     = op_clip;
    g_op_registry[OP_Cos]      = op_cos;
    g_op_registry[OP_Cosh]     = op_cosh;
    g_op_registry[OP_Div]      = op_div;
    g_op_registry[OP_Equal]    = op_equal;
    g_op_registry[OP_Erf]      = op_erf;
    g_op_registry[OP_Exp]      = op_exp;
    g_op_registry[OP_Floor]    = op_floor;
    g_op_registry[OP_Log]      = op_log;
    g_op_registry[OP_Mul]      = op_mul;
    g_op_registry[OP_Neg]      = op_neg;
    g_op_registry[OP_Pow]      = op_pow;
    g_op_registry[OP_Reciprocal] = op_reciprocal;
    g_op_registry[OP_Round]    = op_round;
    g_op_registry[OP_Sign]     = op_sign;
    g_op_registry[OP_Sin]      = op_sin;
    g_op_registry[OP_Sinh]     = op_sinh;
    g_op_registry[OP_Sqrt]     = op_sqrt;
    g_op_registry[OP_Sub]      = op_sub;
    g_op_registry[OP_Tan]      = op_tan;
    g_op_registry[OP_Tanh]     = op_tanh;

    // ========== 卷积类 ==========
    g_op_registry[OP_Conv]     = op_conv2d;

    // ========== 池化类 ==========
    g_op_registry[OP_AveragePool]      = op_avgpool;
    g_op_registry[OP_GlobalAveragePool]= op_global_avg_pool;
    g_op_registry[OP_GlobalMaxPool]    = op_global_max_pool;
    g_op_registry[OP_MaxPool]          = op_maxpool;

    // ========== 归一化类 ==========
    g_op_registry[OP_BatchNormalization]   = op_batch_norm;
    g_op_registry[OP_InstanceNormalization]= op_instance_norm;
    g_op_registry[OP_LayerNormalization]   = op_layer_norm;

    // ========== 激活函数 ==========
    g_op_registry[OP_Celu]         = op_celu;
    g_op_registry[OP_Elu]          = op_elu;
    g_op_registry[OP_Gelu]         = op_gelu;
    g_op_registry[OP_HardSigmoid]  = op_hard_sigmoid;
    g_op_registry[OP_HardSwish]    = op_hard_swish;
    g_op_registry[OP_LeakyRelu]    = op_leaky_relu;
    g_op_registry[OP_PRelu]        = op_prelu;
    g_op_registry[OP_Relu]         = op_relu;
    g_op_registry[OP_Selu]         = op_selu;
    g_op_registry[OP_Sigmoid]      = op_sigmoid;
    g_op_registry[OP_Softmax]      = op_softmax;

    // ========== 矩阵运算 ==========
    g_op_registry[OP_Gemm]     = op_gemm;
    g_op_registry[OP_MatMul]   = op_matmul;

    // ========== Tensor 操作 ==========
    g_op_registry[OP_Cast]     = op_cast;
    g_op_registry[OP_Concat]   = op_concat;
    g_op_registry[OP_Flatten]  = op_flatten;
    g_op_registry[OP_Gather]   = op_gather;
    g_op_registry[OP_Identity] = op_identity;
    g_op_registry[OP_Reshape]  = op_reshape;
    g_op_registry[OP_Split]    = op_split;
    g_op_registry[OP_Squeeze]  = op_squeeze;
    g_op_registry[OP_Transpose]= op_transpose;
    g_op_registry[OP_Unsqueeze]= op_unsqueeze;

    // ========== Reduce 类 ==========
    g_op_registry[OP_ReduceMean] = op_reduce_mean;

// 注册 Cumsum
    g_op_registry[OP_CumSum] = op_cumsum;

    // 注册 ConvTranspose
    g_op_registry[OP_ConvTranspose] = op_conv_transpose;
    
    // 注册 Tensor 其他操作
    g_op_registry[OP_Compress] = op_compress;
    g_op_registry[OP_Constant] = op_constant;
    g_op_registry[OP_ConstantOfShape] = op_constant_of_shape;
    g_op_registry[OP_DepthToSpace] = op_depth_to_space;
    g_op_registry[OP_Dropout] = op_dropout;
    g_op_registry[OP_Einsum] = op_einsum;
    // g_op_registry[OP_EmbeddingBag] = op_embedding_bag; // Non-standard ONNX
    g_op_registry[OP_Expand] = op_expand;
    g_op_registry[OP_EyeLike] = op_eyelike;
    g_op_registry[OP_GatherElements] = op_gather_elements;
    g_op_registry[OP_GatherND] = op_gather_nd;
    
    // 注册其他算子
    g_op_registry[OP_Det] = op_det;
    g_op_registry[OP_DeformConv] = op_deformable_conv;
    g_op_registry[OP_GlobalLpPool] = op_global_lp_pool;
    g_op_registry[OP_Greater] = op_greater;
    g_op_registry[OP_GreaterOrEqual] = op_greater_or_equal;
    g_op_registry[OP_GridSample] = op_grid_sample;
    g_op_registry[OP_GRU] = op_gru;
    g_op_registry[OP_Hardmax] = op_hardmax;
    g_op_registry[OP_If] = op_if;
    g_op_registry[OP_IsInf] = op_is_inf;
    g_op_registry[OP_IsNaN] = op_is_nan;
    g_op_registry[OP_Less] = op_less;
    g_op_registry[OP_LessOrEqual] = op_less_or_equal;
    g_op_registry[OP_LogSoftmax] = op_log_softmax;
    g_op_registry[OP_LpNormalization] = op_lp_normalization;
    g_op_registry[OP_LpPool] = op_lp_pool;
    g_op_registry[OP_LRN] = op_lrn;
    g_op_registry[OP_LSTM] = op_lstm;
    g_op_registry[OP_Max] = op_max;
    g_op_registry[OP_MaxRoiPool] = op_max_roi_pool;
    g_op_registry[OP_MaxUnpool] = op_max_unpool;
    g_op_registry[OP_Mean] = op_mean;
    g_op_registry[OP_MeanVarianceNormalization] = op_mean_variance_norm;
    g_op_registry[OP_Min] = op_min;
    g_op_registry[OP_Mod] = op_mod;
    g_op_registry[OP_Multinomial] = op_multinomial;
    g_op_registry[OP_NonMaxSuppression] = op_non_max_suppression;
    g_op_registry[OP_NonZero] = op_nonzero;
    g_op_registry[OP_Not] = op_not;
    g_op_registry[OP_OneHot] = op_onehot;
    g_op_registry[OP_Or] = op_or;
    g_op_registry[OP_Pad] = op_pad;
    g_op_registry[OP_RandomNormal] = op_random_normal;
    g_op_registry[OP_RandomNormalLike] = op_random_normal_like;
    g_op_registry[OP_RandomUniform] = op_random_uniform;
    g_op_registry[OP_RandomUniformLike] = op_random_uniform_like;
    g_op_registry[OP_Range] = op_range;
    g_op_registry[OP_ReduceL1] = op_reduce_l1;
    g_op_registry[OP_ReduceL2] = op_reduce_l2;
    g_op_registry[OP_ReduceLogSum] = op_reduce_log_sum;
    g_op_registry[OP_ReduceLogSumExp] = op_reduce_log_sum_exp;
    g_op_registry[OP_ReduceMax] = op_reduce_max;
    g_op_registry[OP_ReduceMin] = op_reduce_min;
    g_op_registry[OP_ReduceProd] = op_reduce_prod;
    g_op_registry[OP_ReduceSum] = op_reduce_sum;
    g_op_registry[OP_ReduceSumSquare] = op_reduce_sum_square;
    g_op_registry[OP_Resize] = op_resize;
    g_op_registry[OP_ReverseSequence] = op_reverse_sequence;
    g_op_registry[OP_RoiAlign] = op_roi_align;
    g_op_registry[OP_Scatter] = op_scatter;
    g_op_registry[OP_ScatterElements] = op_scatter_elements;
    g_op_registry[OP_ScatterND] = op_scatter_nd;
    g_op_registry[OP_Shape] = op_shape;
    g_op_registry[OP_Shrink] = op_shrink;
    g_op_registry[OP_Size] = op_size;
    g_op_registry[OP_Slice] = op_slice;
    g_op_registry[OP_SoftmaxCrossEntropyLoss] = op_softmax_cross_entropy_loss;
    g_op_registry[OP_Softplus] = op_softplus;
    g_op_registry[OP_Softsign] = op_softsign;
    g_op_registry[OP_SpaceToDepth] = op_space_to_depth;
    g_op_registry[OP_Sum] = op_sum;
    g_op_registry[OP_TfIdfVectorizer] = op_tfidf_vectorizer;
    g_op_registry[OP_ThresholdedRelu] = op_thresholded_relu;
    g_op_registry[OP_Tile] = op_tile;
    g_op_registry[OP_TopK] = op_topk;
    g_op_registry[OP_Trilu] = op_trilu;
    g_op_registry[OP_Unsqueeze] = op_unsqueeze;
    g_op_registry[OP_Where] = op_where;
    g_op_registry[OP_Xor] = op_xor;
    
    // Hack: YOLOv10n.ennf uses 214 (Optional) for Unsqueeze?
    // Mapping 214 to op_unsqueeze just in case.
    g_op_registry[214] = op_unsqueeze;
    
    g_registry_initialized = 1;
}

/* ============================================================
 * 调度器
 * ============================================================ */
int spinn_dispatch_op(uint16_t op_type,
                      SpinnTensor **inputs, int n_in,
                      void *params, uint16_t params_size,
                      SpinnTensor **outputs, int n_out) {
    init_registry();
    
    if (op_type >= 512) {
        printf("Warning: Op type %d out of range\n", op_type);
        return -1;
    }
    
    SpinnOpKernel kernel = g_op_registry[op_type];
    if (kernel) {
        return kernel(inputs, n_in, params, params_size, outputs, n_out);
    } else {
        printf("  -> Op %d not implemented (skip)\n", op_type);
        return 0; // 跳过未实现的算子
    }
}
