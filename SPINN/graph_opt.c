/*
 * graph_opt.c - ENNF 图优化 Pass 实现
 * 
 * 实现 Conv+BN 融合 和 Conv+ReLU 融合：
 *   - 参照 ONNX Runtime 的 ConvBNFusion 和 ConvActivationFusion
 *   - BN 融合：将 BN 的 scale/bias/mean/var 吸收进 Conv 的 weight 和 bias
 *   - ReLU 融合：标记 Conv 节点带有激活函数，跳过独立的 ReLU 节点
 */

#include "graph_opt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================
 * 辅助函数
 * ============================================================ */

/* 查找某个 tensor name 是否是 initializer (权重) */
static Onnx__TensorProto *find_initializer(Onnx__GraphProto *graph, const char *name) {
    if (!name) return NULL;
    for (size_t i = 0; i < graph->n_initializer; i++) {
        if (graph->initializer[i]->name && strcmp(graph->initializer[i]->name, name) == 0) {
            return graph->initializer[i];
        }
    }
    return NULL;
}

/* 获取 initializer 的 float 数据指针和元素数量 */
static float *get_float_data(Onnx__TensorProto *proto, size_t *n_elements) {
    if (!proto) return NULL;
    
    /* 计算元素数量 */
    size_t count = 1;
    for (size_t i = 0; i < proto->n_dims; i++) count *= proto->dims[i];
    *n_elements = count;
    
    if (proto->raw_data.len > 0) {
        return (float *)proto->raw_data.data;
    } else if (proto->n_float_data > 0) {
        return proto->float_data;
    }
    return NULL;
}

/* 查找某个 tensor output 被多少个后续节点消费 */
static int count_consumers(Onnx__GraphProto *graph, const char *tensor_name,
                           int start_node, uint8_t *node_flags) {
    int count = 0;
    for (size_t i = start_node; i < graph->n_node; i++) {
        if (node_flags[i] & NODE_FLAG_SKIP) continue;
        for (size_t j = 0; j < graph->node[i]->n_input; j++) {
            if (graph->node[i]->input[j] && strcmp(graph->node[i]->input[j], tensor_name) == 0) {
                count++;
            }
        }
    }
    /* 同时检查是否是图的输出 */
    for (size_t i = 0; i < graph->n_output; i++) {
        if (graph->output[i]->name && strcmp(graph->output[i]->name, tensor_name) == 0) {
            count++;
        }
    }
    return count;
}

/* 获取节点的属性值 */
static Onnx__AttributeProto *get_attr(Onnx__NodeProto *node, const char *name) {
    for (size_t i = 0; i < node->n_attribute; i++) {
        if (strcmp(node->attribute[i]->name, name) == 0) {
            return node->attribute[i];
        }
    }
    return NULL;
}

/* ============================================================
 * Pass 1: Conv + BatchNorm 融合
 * 
 * 数学原理：
 *   BN(x) = gamma * (x - mean) / sqrt(var + eps) + beta
 *   
 *   令 Conv 输出 y = W*x + b，则：
 *   BN(y) = gamma * (W*x + b - mean) / sqrt(var+eps) + beta
 *         = (gamma/sqrt(var+eps)) * W * x + (gamma*(b-mean)/sqrt(var+eps) + beta)
 *   
 *   融合后的新参数：
 *   W_new = W * (gamma / sqrt(var + eps))       (乘以每OC通道的缩放因子)
 *   b_new = gamma * (b - mean) / sqrt(var+eps) + beta
 *
 * 参照 ORT: onnxruntime/core/optimizer/conv_bn_fusion.cc
 * ============================================================ */
static int fuse_conv_bn(GraphOptContext *ctx) {
    Onnx__GraphProto *graph = ctx->graph;
    int fused = 0;
    
    for (size_t i = 0; i + 1 < graph->n_node; i++) {
        if (ctx->node_flags[i] & NODE_FLAG_SKIP) continue;
        
        Onnx__NodeProto *conv = graph->node[i];
        
        /* 检查是否为 Conv 节点 */
        if (strcmp(conv->op_type, "Conv") != 0) continue;
        if (conv->n_output < 1 || !conv->output[0]) continue;
        
        /* Conv 的输出只能有一个消费者（BN），否则不能融合 */
        const char *conv_output = conv->output[0];
        if (count_consumers(graph, conv_output, i + 1, ctx->node_flags) != 1) continue;
        
        /* 找到 Conv 的下一个消费者，检查是否为 BN */
        Onnx__NodeProto *bn = NULL;
        size_t bn_idx = 0;
        for (size_t j = i + 1; j < graph->n_node; j++) {
            if (ctx->node_flags[j] & NODE_FLAG_SKIP) continue;
            if (graph->node[j]->n_input > 0 &&
                graph->node[j]->input[0] &&
                strcmp(graph->node[j]->input[0], conv_output) == 0) {
                if (strcmp(graph->node[j]->op_type, "BatchNormalization") == 0) {
                    bn = graph->node[j];
                    bn_idx = j;
                }
                break;
            }
        }
        if (!bn) continue;
        
        /* BN 需要 5 个输入：X, scale, B, mean, var */
        if (bn->n_input < 5) continue;
        
        /* 获取 BN 参数 */
        size_t n_scale = 0, n_bias = 0, n_mean = 0, n_var = 0;
        float *bn_scale = get_float_data(find_initializer(graph, bn->input[1]), &n_scale);
        float *bn_bias  = get_float_data(find_initializer(graph, bn->input[2]), &n_bias);
        float *bn_mean  = get_float_data(find_initializer(graph, bn->input[3]), &n_mean);
        float *bn_var   = get_float_data(find_initializer(graph, bn->input[4]), &n_var);
        
        if (!bn_scale || !bn_bias || !bn_mean || !bn_var) {
            printf("  [SKIP] Conv+BN @node %zu: BN params not all available as initializers\n", i);
            continue;
        }
        
        /* 获取 BN epsilon */
        float eps = 1e-5f;
        Onnx__AttributeProto *eps_attr = get_attr(bn, "epsilon");
        if (eps_attr) eps = eps_attr->f;
        
        /* 获取 Conv weight */
        if (conv->n_input < 2) continue;
        Onnx__TensorProto *w_proto = find_initializer(graph, conv->input[1]);
        if (!w_proto) {
            printf("  [SKIP] Conv+BN @node %zu: Conv weight not an initializer\n", i);
            continue;
        }
        
        size_t n_w;
        float *w_data = get_float_data(w_proto, &n_w);
        if (!w_data) continue;
        
        int OC = (w_proto->n_dims > 0) ? (int)w_proto->dims[0] : 0;
        if (OC <= 0 || (int)n_scale != OC) continue;
        int w_per_oc = (int)(n_w / OC);
        
        /* 获取或创建 Conv bias */
        float *conv_bias = NULL;
        Onnx__TensorProto *b_proto = NULL;
        size_t n_bias_orig = 0;
        int has_bias = (conv->n_input >= 3 && conv->input[2] && strlen(conv->input[2]) > 0);
        
        if (has_bias) {
            b_proto = find_initializer(graph, conv->input[2]);
            if (b_proto) conv_bias = get_float_data(b_proto, &n_bias_orig);
        }
        
        /* =============================================
         * 执行融合：原地修改 Conv 的 weight 和 bias
         * ============================================= */
        printf("  [FUSE] Conv+BN: node %zu (Conv) + node %zu (BN) -> fused Conv\n", i, bn_idx);
        
        /* 计算融合系数并原地修改 weight */
        for (int oc = 0; oc < OC; oc++) {
            float factor = bn_scale[oc] / sqrtf(bn_var[oc] + eps);
            
            /* W_new[oc] = W[oc] * factor */
            float *w_oc = w_data + oc * w_per_oc;
            for (int j = 0; j < w_per_oc; j++) {
                w_oc[j] *= factor;
            }
        }
        
        /* 计算融合 bias (写入 BN 的 bias initializer，复用其存储) */
        /* b_new[oc] = factor * (b_old[oc] - mean[oc]) + bn_bias[oc] */
        for (int oc = 0; oc < OC; oc++) {
            float factor = bn_scale[oc] / sqrtf(bn_var[oc] + eps);
            float b_old = (conv_bias && oc < (int)n_bias_orig) ? conv_bias[oc] : 0.0f;
            bn_bias[oc] = factor * (b_old - bn_mean[oc]) + bn_bias[oc];
        }
        
        /* 修改 Conv 节点：
         * - 输出改为 BN 的输出（跳过 BN）
         * - 如果原来没 bias，则 bias 指向 BN 的 bias initializer
         */
        free(conv->output[0]);
        conv->output[0] = strdup(bn->output[0]);
        
        if (!has_bias) {
            /* 扩展 Conv 的 input 为 3 个：X, W, B */
            conv->n_input = 3;
            conv->input = realloc(conv->input, 3 * sizeof(char*));
            conv->input[2] = strdup(bn->input[2]); /* BN 的 bias tensor */
        } else {
            /* 把融合后的 bias 写回 Conv 的 bias（已经原地修改了 bn_bias） */
            /* Conv 的 bias 指向 BN 的 bias initializer */
            free(conv->input[2]);
            conv->input[2] = strdup(bn->input[2]);
        }
        
        /* 标记 BN 节点为跳过 */
        ctx->node_flags[bn_idx] = NODE_FLAG_SKIP;
        fused++;
    }
    
    return fused;
}

/* ============================================================
 * Pass 2: Conv/Gemm + ReLU 融合
 * 
 * 在 Conv 的输出后紧接 ReLU 时，标记 Conv 节点的 fused_act = RELU，
 * 运行时在 Conv 内核输出时直接 clamp 到 0，跳过独立的 ReLU 节点。
 *
 * 参照 ORT: ConvActivationFusion
 * ============================================================ */
static int fuse_conv_relu(GraphOptContext *ctx) {
    Onnx__GraphProto *graph = ctx->graph;
    int fused = 0;
    
    for (size_t i = 0; i + 1 < graph->n_node; i++) {
        if (ctx->node_flags[i] & NODE_FLAG_SKIP) continue;
        
        Onnx__NodeProto *conv = graph->node[i];
        
        /* 检查是否为 Conv 或 Gemm */
        int is_conv = (strcmp(conv->op_type, "Conv") == 0);
        /* 暂时只对 Conv 做 */
        if (!is_conv) continue;
        if (conv->n_output < 1 || !conv->output[0]) continue;
        
        /* Conv 的输出只能有一个消费者 */
        const char *conv_output = conv->output[0];
        if (count_consumers(graph, conv_output, i + 1, ctx->node_flags) != 1) continue;
        
        /* 找到 Conv 的下一个消费者，检查是否为 ReLU */
        Onnx__NodeProto *relu = NULL;
        size_t relu_idx = 0;
        for (size_t j = i + 1; j < graph->n_node; j++) {
            if (ctx->node_flags[j] & NODE_FLAG_SKIP) continue;
            if (graph->node[j]->n_input > 0 &&
                graph->node[j]->input[0] &&
                strcmp(graph->node[j]->input[0], conv_output) == 0) {
                if (strcmp(graph->node[j]->op_type, "Relu") == 0) {
                    relu = graph->node[j];
                    relu_idx = j;
                }
                break;
            }
        }
        if (!relu) continue;
        
        /* 执行融合 */
        printf("  [FUSE] Conv+ReLU: node %zu (Conv) + node %zu (Relu) -> fused Conv+ReLU\n", i, relu_idx);
        
        /* Conv 的输出改为 ReLU 的输出 */
        free(conv->output[0]);
        conv->output[0] = strdup(relu->output[0]);
        
        /* 标记激活类型 */
        ctx->fused_act[i] = FUSED_ACT_RELU;
        ctx->node_flags[i] |= NODE_FLAG_HAS_ACT;
        
        /* 标记 ReLU 节点为跳过 */
        ctx->node_flags[relu_idx] = NODE_FLAG_SKIP;
        fused++;
    }
    
    return fused;
}

/* ============================================================
 * 公开 API
 * ============================================================ */

GraphOptContext *graph_opt_init(Onnx__GraphProto *graph) {
    GraphOptContext *ctx = calloc(1, sizeof(GraphOptContext));
    if (!ctx) return NULL;
    
    ctx->graph = graph;
    ctx->node_count = graph->n_node;
    ctx->node_flags = calloc(graph->n_node, sizeof(uint8_t));
    ctx->fused_act = calloc(graph->n_node, sizeof(uint8_t));
    ctx->fusions_applied = 0;
    
    return ctx;
}

int graph_opt_run(GraphOptContext *ctx) {
    printf("\n=== Graph Optimization ===\n");
    printf("Total nodes before optimization: %d\n", ctx->node_count);
    
    /* Pass 1: Conv + BN fusion (多次迭代直到无更多融合) */
    int total_conv_bn = 0;
    for (int iter = 0; iter < 10; iter++) {
        int n = fuse_conv_bn(ctx);
        total_conv_bn += n;
        if (n == 0) break;
    }
    printf("Conv+BN fusions: %d\n", total_conv_bn);
    
    /* Pass 2: Conv + ReLU fusion (在 BN 融合之后做) */
    int total_conv_relu = fuse_conv_relu(ctx);
    printf("Conv+ReLU fusions: %d\n", total_conv_relu);
    
    /* 统计 */
    int skipped = 0;
    for (int i = 0; i < ctx->node_count; i++) {
        if (ctx->node_flags[i] & NODE_FLAG_SKIP) skipped++;
    }
    printf("Nodes after optimization: %d (skipped %d)\n", 
           ctx->node_count - skipped, skipped);
    printf("=========================\n\n");
    
    ctx->fusions_applied = total_conv_bn + total_conv_relu;
    return ctx->fusions_applied;
}

void graph_opt_free(GraphOptContext *ctx) {
    if (!ctx) return;
    free(ctx->node_flags);
    free(ctx->fused_act);
    free(ctx);
}
