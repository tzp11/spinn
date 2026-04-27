/*
 * onnx2ennf_ops.c - 完整算子参数提取实现
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "ennf_op_types.h"
#include "ennf_op_params.h"
#include "onnx.proto3.pb-c.h"

// ======================================================================================
// 辅助函数
// ======================================================================================

static int64_t get_attr_int(Onnx__NodeProto *node, const char *name, int64_t def) {
    for (int i = 0; i < node->n_attribute; i++) {
        if (strcmp(node->attribute[i]->name, name) == 0) {
            return node->attribute[i]->i;
        }
    }
    return def;
}

static float get_attr_float(Onnx__NodeProto *node, const char *name, float def) {
    for (int i = 0; i < node->n_attribute; i++) {
        if (strcmp(node->attribute[i]->name, name) == 0) {
            return node->attribute[i]->f;
        }
    }
    return def;
}

static void extract_int_array_u16(Onnx__NodeProto *node, const char *name, 
                                 uint16_t *out, int max_len) {
    for (int i = 0; i < node->n_attribute; i++) {
        if (strcmp(node->attribute[i]->name, name) == 0) {
            int n = node->attribute[i]->n_ints;
            for (int j = 0; j < n && j < max_len; j++) {
                out[j] = (uint16_t)node->attribute[i]->ints[j];
            }
            return;
        }
    }
}

static void extract_int_array_i32(Onnx__NodeProto *node, const char *name, 
                                 int32_t *out, int max_len) {
    for (int i = 0; i < node->n_attribute; i++) {
        if (strcmp(node->attribute[i]->name, name) == 0) {
            int n = node->attribute[i]->n_ints;
            for (int j = 0; j < n && j < max_len; j++) {
                out[j] = (int32_t)node->attribute[i]->ints[j];
            }
            return;
        }
    }
}

static void extract_int_array_i32_with_count(Onnx__NodeProto *node, const char *name, 
                                 int32_t *out, uint8_t *count_out, int max_len) {
    for (int i = 0; i < node->n_attribute; i++) {
        if (strcmp(node->attribute[i]->name, name) == 0) {
            int n = node->attribute[i]->n_ints;
            if (n > max_len) n = max_len;
            if (count_out) *count_out = (uint8_t)n;
            for (int j = 0; j < n; j++) {
                out[j] = (int32_t)node->attribute[i]->ints[j];
            }
            return;
        }
    }
}

static uint8_t get_attr_auto_pad(Onnx__NodeProto *node) {
    for (int i = 0; i < node->n_attribute; i++) {
        if (strcmp(node->attribute[i]->name, "auto_pad") == 0) {
            const char *s = (char*)node->attribute[i]->s.data;
            if (strcmp(s, "SAME_UPPER") == 0) return ENNF_AUTOPAD_SAME_UPPER;
            if (strcmp(s, "SAME_LOWER") == 0) return ENNF_AUTOPAD_SAME_LOWER;
            if (strcmp(s, "VALID") == 0) return ENNF_AUTOPAD_VALID;
            return ENNF_AUTOPAD_NOTSET;
        }
    }
    return ENNF_AUTOPAD_NOTSET; 
}

// ======================================================================================
// 主提取函数
// ======================================================================================
void extract_op_params(Onnx__NodeProto *node, ENNF_OpType type, 
                       uint8_t *buf, uint16_t *size) {
    *size = 0;
    
    switch (type) {
        /* ===== 卷积类 ===== */
        case OP_Conv:
        case OP_QLinearConv: {
            ENNF_ConvParams *p = (ENNF_ConvParams*)buf;
            memset(p, 0, sizeof(*p));
            extract_int_array_u16(node, "kernel_shape", p->kernel_shape, 4);
            extract_int_array_u16(node, "strides", p->strides, 4);
            extract_int_array_u16(node, "pads", p->pads, 8);
            extract_int_array_u16(node, "dilations", p->dilations, 4);
            p->group = (uint16_t)get_attr_int(node, "group", 1);
            p->auto_pad = get_attr_auto_pad(node);
            *size = sizeof(ENNF_ConvParams);
            break;
        }
        case OP_ConvTranspose: {
            ENNF_ConvTransposeParams *p = (ENNF_ConvTransposeParams*)buf;
            memset(p, 0, sizeof(*p));
            extract_int_array_u16(node, "kernel_shape", p->base.kernel_shape, 4);
            extract_int_array_u16(node, "strides", p->base.strides, 4);
            extract_int_array_u16(node, "pads", p->base.pads, 8);
            extract_int_array_u16(node, "dilations", p->base.dilations, 4);
            p->base.group = (uint16_t)get_attr_int(node, "group", 1);
            p->base.auto_pad = get_attr_auto_pad(node);
            extract_int_array_u16(node, "output_padding", p->output_padding, 4);
            extract_int_array_u16(node, "output_shape", p->output_shape, 4);
            *size = sizeof(ENNF_ConvTransposeParams);
            break;
        }
        
        /* ===== 池化类 ===== */
        case OP_MaxPool:
        case OP_AveragePool:
        case OP_LpPool: {
            ENNF_PoolParams *p = (ENNF_PoolParams*)buf;
            memset(p, 0, sizeof(*p));
            extract_int_array_u16(node, "kernel_shape", p->kernel_shape, 4);
            extract_int_array_u16(node, "strides", p->strides, 4);
            extract_int_array_u16(node, "pads", p->pads, 8);
            extract_int_array_u16(node, "dilations", p->dilations, 4);
            p->auto_pad = get_attr_auto_pad(node);
            p->ceil_mode = (uint8_t)get_attr_int(node, "ceil_mode", 0);
            p->count_include_pad = (uint8_t)get_attr_int(node, "count_include_pad", 0);
            p->storage_order = (uint8_t)get_attr_int(node, "storage_order", 0);
            if (type == OP_LpPool) p->p = (int32_t)get_attr_int(node, "p", 2);
            *size = sizeof(ENNF_PoolParams);
            break;
        }
        
        /* ===== 矩阵运算 ===== */
        case OP_Gemm: {
            ENNF_GemmParams *p = (ENNF_GemmParams*)buf;
            memset(p, 0, sizeof(*p));
            p->alpha = get_attr_float(node, "alpha", 1.0f);
            p->beta = get_attr_float(node, "beta", 1.0f);
            p->transA = (uint8_t)get_attr_int(node, "transA", 0);
            p->transB = (uint8_t)get_attr_int(node, "transB", 0);
            *size = sizeof(ENNF_GemmParams);
            break;
        }
        
        /* ===== 归一化类 ===== */
        case OP_BatchNormalization: {
            ENNF_BatchNormParams *p = (ENNF_BatchNormParams*)buf;
            memset(p, 0, sizeof(*p));
            p->epsilon = get_attr_float(node, "epsilon", 1e-5f);
            p->momentum = get_attr_float(node, "momentum", 0.9f);
            p->training_mode = (uint8_t)get_attr_int(node, "training_mode", 0);
            *size = sizeof(ENNF_BatchNormParams);
            break;
        }
        case OP_LayerNormalization:
        case OP_GroupNormalization: {
            ENNF_LayerNormParams *p = (ENNF_LayerNormParams*)buf;
            memset(p, 0, sizeof(*p));
            p->axis = (int32_t)get_attr_int(node, "axis", -1);
            p->epsilon = get_attr_float(node, "epsilon", 1e-5f);
            p->stash_type = (uint8_t)get_attr_int(node, "stash_type", 1);
            if(type == OP_GroupNormalization) p->num_groups = (int32_t)get_attr_int(node, "num_groups", 1);
            *size = sizeof(ENNF_LayerNormParams);
            break;
        }
        case OP_LRN: {
             ENNF_LRNParams *p = (ENNF_LRNParams*)buf;
             p->alpha = get_attr_float(node, "alpha", 0.0001f);
             p->beta = get_attr_float(node, "beta", 0.75f);
             p->bias = get_attr_float(node, "bias", 1.0f);
             p->size = (int32_t)get_attr_int(node, "size", 1);
             *size = sizeof(ENNF_LRNParams);
             break;
        }
        
        /* ===== 激活类 ===== */
        case OP_LeakyRelu: {
            ENNF_LeakyReluParams *p = (ENNF_LeakyReluParams*)buf;
            p->alpha = get_attr_float(node, "alpha", 0.01f);
            *size = sizeof(ENNF_LeakyReluParams);
            break;
        }
        case OP_Elu:
        case OP_Celu: {
            ENNF_EluParams *p = (ENNF_EluParams*)buf;
            p->alpha = get_attr_float(node, "alpha", 1.0f);
            // Selu has gamma too
            *size = sizeof(ENNF_EluParams);
            break;
        }
        case OP_Selu: {
            ENNF_EluParams *p = (ENNF_EluParams*)buf;
            p->alpha = get_attr_float(node, "alpha", 1.67326f);
            p->gamma = get_attr_float(node, "gamma", 1.0507f);
            *size = sizeof(ENNF_EluParams);
            break;
        }
        case OP_Clip: {
            ENNF_ClipParams *p = (ENNF_ClipParams*)buf;
            // Opset < 11 used attributes. Later used inputs.
            p->min_val = get_attr_float(node, "min", -3.40282e+38f);
            p->max_val = get_attr_float(node, "max", 3.40282e+38f);
            *size = sizeof(ENNF_ClipParams);
            break;
        }
        case OP_HardSigmoid:
        case OP_HardSwish: {
             ENNF_HardSigmoidParams *p = (ENNF_HardSigmoidParams*)buf;
             p->alpha = get_attr_float(node, "alpha", 0.2f);
             p->beta = get_attr_float(node, "beta", 0.5f);
             *size = sizeof(ENNF_HardSigmoidParams);
             break;
        }
        case OP_Softmax: {
             ENNF_SoftmaxParams *p = (ENNF_SoftmaxParams*)buf;
             p->axis = (int32_t)get_attr_int(node, "axis", -1);
             *size = sizeof(ENNF_SoftmaxParams);
             break;
        }
        case OP_LogSoftmax: {
             ENNF_SoftmaxParams *p = (ENNF_SoftmaxParams*)buf;
             p->axis = (int32_t)get_attr_int(node, "axis", -1);
             *size = sizeof(ENNF_SoftmaxParams);
             break;
        }
        
        /* ===== Tensor 操作 ===== */
        case OP_Transpose: {
            ENNF_TransposeParams *p = (ENNF_TransposeParams*)buf;
            memset(p, 0, sizeof(*p));
            //perm attribute
            for (int i = 0; i < node->n_attribute; i++) {
                if (strcmp(node->attribute[i]->name, "perm") == 0) {
                    int n = node->attribute[i]->n_ints;
                    p->num_dims = n;
                    for(int j=0; j<n && j<ENNF_MAX_DIMS; j++) {
                        p->perm[j] = (uint8_t)node->attribute[i]->ints[j];
                    }
                }
            }
            *size = sizeof(ENNF_TransposeParams);
            break;
        }
        case OP_Flatten: {
            ENNF_FlattenParams *p = (ENNF_FlattenParams*)buf;
            p->axis = (int32_t)get_attr_int(node, "axis", 1);
            *size = sizeof(ENNF_FlattenParams);
            break;
        }
        case OP_Squeeze: 
        case OP_Unsqueeze: {
             ENNF_SqueezeParams *p = (ENNF_SqueezeParams*)buf;
             memset(p, 0, sizeof(*p));
             extract_int_array_i32_with_count(node, "axes", p->axes, &p->num_axes, ENNF_MAX_DIMS);
             *size = sizeof(ENNF_SqueezeParams);
             break;
        }
        case OP_Concat: {
            ENNF_ConcatParams *p = (ENNF_ConcatParams*)buf;
            p->axis = (int32_t)get_attr_int(node, "axis", 1);
            *size = sizeof(ENNF_ConcatParams);
            break;
        }
        case OP_Split: {
            ENNF_SplitParams *p = (ENNF_SplitParams*)buf;
            p->axis = (int32_t)get_attr_int(node, "axis", 0);
            // splits attribute exists in older opsets.
            *size = sizeof(ENNF_SplitParams);
            break;
        }
        case OP_Pad: {
            ENNF_PadParams *p = (ENNF_PadParams*)buf;
            memset(p, 0, sizeof(*p));
            char *mode = "constant";
            for (int i = 0; i < node->n_attribute; i++) {
                if (strcmp(node->attribute[i]->name, "mode") == 0) {
                    mode = (char*)node->attribute[i]->s.data;
                }
            }
            if (strcmp(mode, "reflect")==0) p->mode = 1;
            else if (strcmp(mode, "edge")==0) p->mode = 2;
            else if (strcmp(mode, "wrap")==0) p->mode = 3;
            else p->mode = 0;
            
            p->constant_value = get_attr_float(node, "value", 0.0f); // only for older opsets
            *size = sizeof(ENNF_PadParams);
            break;
        }
        case OP_Gather:
        case OP_GatherElements: {
            ENNF_GatherParams *p = (ENNF_GatherParams*)buf;
            p->axis = (int32_t)get_attr_int(node, "axis", 0);
            *size = sizeof(ENNF_GatherParams);
            break;
        }
        case OP_GatherND: {
            ENNF_GatherParams *p = (ENNF_GatherParams*)buf;
            p->batch_dims = (int32_t)get_attr_int(node, "batch_dims", 0);
            *size = sizeof(ENNF_GatherParams);
            break;
        }
        case OP_Scatter:
        case OP_ScatterElements:
        case OP_ScatterND: {
            ENNF_ScatterParams *p = (ENNF_ScatterParams*)buf;
            p->axis = (int32_t)get_attr_int(node, "axis", 0);
            char *red = "none";
            // Check attribute 'reduction'
             for (int i = 0; i < node->n_attribute; i++) {
                if (strcmp(node->attribute[i]->name, "reduction") == 0) {
                    red = (char*)node->attribute[i]->s.data;
                }
            }
            if (strcmp(red, "add")==0) p->reduction = 1;
            else if (strcmp(red, "mul")==0) p->reduction = 2;
            // ...
            *size = sizeof(ENNF_ScatterParams);
            break;
        }
        case OP_TopK: {
            ENNF_TopKParams *p = (ENNF_TopKParams*)buf;
            p->axis = (int32_t)get_attr_int(node, "axis", -1);
            p->largest = (uint8_t)get_attr_int(node, "largest", 1);
            p->sorted = (uint8_t)get_attr_int(node, "sorted", 1);
            *size = sizeof(ENNF_TopKParams);
            break;
        }
        case OP_OneHot: {
            ENNF_OneHotParams *p = (ENNF_OneHotParams*)buf;
            p->axis = (int32_t)get_attr_int(node, "axis", -1);
            *size = sizeof(ENNF_OneHotParams);
            break;
        }
        case OP_Cast: {
            ENNF_CastParams *p = (ENNF_CastParams*)buf;
            p->to_type = (uint8_t)get_attr_int(node, "to", 0);
            p->saturate = (uint8_t)get_attr_int(node, "saturate", 1);
            *size = sizeof(ENNF_CastParams);
            break;
        }
        case OP_Trilu: {
            ENNF_TriluParams *p = (ENNF_TriluParams*)buf;
            p->upper = (uint8_t)get_attr_int(node, "upper", 1);
            *size = sizeof(ENNF_TriluParams);
            break;
        }
        
        /* ===== Reduce 类 ===== */
        case OP_ReduceMean:
        case OP_ReduceMax:
        case OP_ReduceMin:
        case OP_ReduceSum:
        case OP_ReduceProd:
        case OP_ReduceL1:
        case OP_ReduceL2:
        case OP_ReduceLogSum:
        case OP_ReduceLogSumExp:
        case OP_ReduceSumSquare: {
            ENNF_ReduceParams *p = (ENNF_ReduceParams*)buf;
            memset(p, 0, sizeof(*p));
            extract_int_array_i32_with_count(node, "axes", p->axes, &p->num_axes, ENNF_MAX_DIMS);
            p->keepdims = (uint8_t)get_attr_int(node, "keepdims", 1);
            p->noop_with_empty_axes = (uint8_t)get_attr_int(node, "noop_with_empty_axes", 0);
            *size = sizeof(ENNF_ReduceParams);
            break;
        }
        
        /* ===== RNN 类 ===== */
        case OP_LSTM: {
            ENNF_LSTMParams *p = (ENNF_LSTMParams*)buf;
            memset(p, 0, sizeof(*p));
            p->hidden_size = get_attr_int(node, "hidden_size", 0);
            char *dir = "forward";
            for (int i = 0; i < node->n_attribute; i++) {
                if (strcmp(node->attribute[i]->name, "direction") == 0) {
                    dir = (char*)node->attribute[i]->s.data;
                }
            }
            if (strcmp(dir, "reverse")==0) p->direction = 1;
            else if (strcmp(dir, "bidirectional")==0) p->direction = 2;
            else p->direction = 0;
            
            p->input_forget = (uint8_t)get_attr_int(node, "input_forget", 0);
            p->layout = (uint8_t)get_attr_int(node, "layout", 0);
            p->clip = get_attr_float(node, "clip", 0.0f); // Default?
            *size = sizeof(ENNF_LSTMParams);
            break;
        }
        case OP_GRU: {
            ENNF_GRUParams *p = (ENNF_GRUParams*)buf;
            memset(p, 0, sizeof(*p));
            p->hidden_size = get_attr_int(node, "hidden_size", 0);
            p->linear_before_reset = (uint8_t)get_attr_int(node, "linear_before_reset", 0);
            // direction ...
            *size = sizeof(ENNF_GRUParams);
            break;
        }
        case OP_RNN: {
            ENNF_RNNParams *p = (ENNF_RNNParams*)buf;
            memset(p, 0, sizeof(*p));
            p->hidden_size = get_attr_int(node, "hidden_size", 0);
            *size = sizeof(ENNF_RNNParams);
            break;
        }
        
        /* ===== 其他参数化算子 ===== */
        case OP_Dropout: {
            ENNF_DropoutParams *p = (ENNF_DropoutParams*)buf;
            p->ratio = get_attr_float(node, "ratio", 0.5f);
            p->seed = get_attr_int(node, "seed", 0);
            *size = sizeof(ENNF_DropoutParams);
            break;
        }
        case OP_Shape: {
            ENNF_ShapeParams *p = (ENNF_ShapeParams*)buf;
            p->start = (int32_t)get_attr_int(node, "start", 0);
            p->end = (int32_t)get_attr_int(node, "end", 0); // optional?
            *size = sizeof(ENNF_ShapeParams);
            break;
        }
        
        /* ===== 默认无参数 ===== */
        default:
            *size = 0;
            break;
    }
}
