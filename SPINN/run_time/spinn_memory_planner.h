/*
 * spinn_memory_planner.h - 静态内存规划器
 * 
 * 功能: 基于 Tensor 生命周期计算 Offset，实现零碎片 Arena 复用
 */

#ifndef __SPINN_MEMORY_PLANNER_H__
#define __SPINN_MEMORY_PLANNER_H__

#include "spinn_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 错误码定义 */
#define SPINN_MEM_OK               0      /* 成功 */
#define SPINN_MEM_ERR_NULL        -1      /* 空指针参数 */
#define SPINN_MEM_ERR_NOMEM       -2      /* 内存不足 (超过 M_max 限制) */
#define SPINN_MEM_ERR_INTERNAL    -3      /* 内部错误 */

/**
 * 执行静态内存规划
 * 
 * 遍历所有节点，模拟执行流程，为每个中间 Tensor 计算 offset。
 * 同时计算 Arena 所需的峰值大小。
 * 
 * @param ctx 已加载的上下文 (tensors 和 nodes 已填充)
 * @param max_arena_bytes Arena 内存上限 (0 表示无限制)
 * @return Arena 所需的总字节数 (>=0 成功), 或 <0 表示错误
 *         SPINN_MEM_ERR_NOMEM: 超过内存上限
 */
int spinn_plan_memory(SpinnContext *ctx, uint32_t max_arena_bytes);

#ifdef __cplusplus
}
#endif

#endif /* __SPINN_MEMORY_PLANNER_H__ */
