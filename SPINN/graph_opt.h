/*
 * graph_opt.h - ENNF 图优化 Pass
 * 
 * 在 onnx2ennf 转换时对 ONNX 图进行优化：
 *   1. Conv + BN Fusion:  BN 参数吸收进 Conv 的 weight/bias
 *   2. Conv + ReLU Fusion: 标记 Conv 节点携带激活函数
 * 
 * 设计原则：
 *   - 只做安全的融合（不改变语义）
 *   - 被融合的节点标记为 skip，不删除（保持节点索引稳定）
 *   - 未能融合的节点保持原样，所有原始算子仍可独立工作
 */

#ifndef __GRAPH_OPT_H__
#define __GRAPH_OPT_H__

#include "onnx.proto3.pb-c.h"
#include <stdint.h>

/* ============================================================
 * 节点状态标记
 * ============================================================ */
#define NODE_FLAG_NONE    0
#define NODE_FLAG_SKIP    1    /* 该节点已被融合到前序节点，转换时跳过 */
#define NODE_FLAG_HAS_ACT 2   /* Conv/Gemm 节点融合了激活函数 */

/* 激活类型 (嵌入 Conv 参数的 reserved 字段) */
#define FUSED_ACT_NONE    0
#define FUSED_ACT_RELU    1
#define FUSED_ACT_CLIP    2   /* ReLU6 等 */

/* ============================================================
 * 图优化上下文
 * ============================================================ */
typedef struct {
    Onnx__GraphProto *graph;
    uint8_t *node_flags;     /* 每个节点的标记，node_count 长度 */
    uint8_t *fused_act;      /* 每个节点的融合激活类型 */
    int node_count;
    int fusions_applied;     /* 统计融合数量 */
} GraphOptContext;

/* ============================================================
 * API
 * ============================================================ */

/* 初始化图优化上下文 */
GraphOptContext *graph_opt_init(Onnx__GraphProto *graph);

/* 执行所有图优化 Pass (按优先级顺序) */
int graph_opt_run(GraphOptContext *ctx);

/* 释放 */
void graph_opt_free(GraphOptContext *ctx);

#endif /* __GRAPH_OPT_H__ */
