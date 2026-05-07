/*
 * spinn_memory_planner.c - 静态内存规划器实现
 * 
 * 核心算法: Interval Packing (区间打包)
 * - 遍历所有节点，模拟执行
 * - 为每个输出 Tensor 寻找最小可用 Offset
 * - 当输入 Tensor 的 ref_count 归零时，释放其占用区间
 */

#include "spinn_memory_planner.h"
#include <stdlib.h>
#include <string.h>

/* ============================================================
 * 内部数据结构: 空闲区间链表
 * ============================================================ */
typedef struct FreeBlock {
    uint32_t offset;
    uint32_t size;
    struct FreeBlock *next;
} FreeBlock;

/* 空闲链表管理 */
static FreeBlock *g_free_list = NULL;

/* 诊断信息：记录分配失败的 tensor */
static uint16_t g_failed_tensor_id = 0;
static uint32_t g_failed_required_size = 0;

static void free_list_init(uint32_t max_size) {
    // 初始化: 一个空闲块从 offset=0 开始
    // max_size=0 表示无限制，使用最大值
    g_free_list = (FreeBlock*)malloc(sizeof(FreeBlock));
    g_free_list->offset = 0;
    g_free_list->size = (max_size > 0) ? max_size : 0xFFFFFFFF;
    g_free_list->next = NULL;
    g_failed_tensor_id = 0;
    g_failed_required_size = 0;
}

static void free_list_destroy(void) {
    FreeBlock *p = g_free_list;
    while (p) {
        FreeBlock *next = p->next;
        free(p);
        p = next;
    }
    g_free_list = NULL;
}

/**
 * 在空闲链表中寻找 Best-Fit 块
 * @param size 所需大小 (对齐后)
 * @return 分配的 offset, 或 0xFFFFFFFF 表示失败
 */
static uint32_t alloc_offset(uint32_t size) {
    if (size == 0) return 0;
    
    // 对齐到 16 字节
    size = (size + 15) & ~15;
    
    FreeBlock *best = NULL;
    FreeBlock *best_prev = NULL;
    FreeBlock *prev = NULL;
    FreeBlock *p = g_free_list;
    
    // Best-Fit: 找最小的能容纳的块
    while (p) {
        if (p->size >= size) {
            if (!best || p->size < best->size) {
                best = p;
                best_prev = prev;
            }
        }
        prev = p;
        p = p->next;
    }
    
    if (!best) return 0xFFFFFFFF; // 无可用空间
    
    uint32_t result = best->offset;
    
    if (best->size == size) {
        // 精确匹配，移除此块
        if (best_prev) {
            best_prev->next = best->next;
        } else {
            g_free_list = best->next;
        }
        free(best);
    } else {
        // 分割
        best->offset += size;
        best->size -= size;
    }
    
    return result;
}

/**
 * 释放一个区间，尝试合并相邻块
 */
static void free_offset(uint32_t offset, uint32_t size) {
    if (size == 0) return;
    size = (size + 15) & ~15;
    
    // 找插入位置 (保持按 offset 升序)
    FreeBlock *prev = NULL;
    FreeBlock *p = g_free_list;
    while (p && p->offset < offset) {
        prev = p;
        p = p->next;
    }
    
    // 创建新块
    FreeBlock *newb = (FreeBlock*)malloc(sizeof(FreeBlock));
    newb->offset = offset;
    newb->size = size;
    newb->next = p;
    
    if (prev) {
        prev->next = newb;
    } else {
        g_free_list = newb;
    }
    
    // 尝试合并: newb 与后继
    if (newb->next && newb->offset + newb->size == newb->next->offset) {
        FreeBlock *merged = newb->next;
        newb->size += merged->size;
        newb->next = merged->next;
        free(merged);
    }
    
    // 尝试合并: prev 与 newb
    if (prev && prev->offset + prev->size == newb->offset) {
        prev->size += newb->size;
        prev->next = newb->next;
        free(newb);
    }
}

/* ============================================================
 * 公共 API 实现
 * ============================================================ */
int spinn_plan_memory(SpinnContext *ctx, uint32_t max_arena_bytes) {
    if (!ctx || !ctx->tensors || !ctx->nodes) return SPINN_MEM_ERR_NULL;
    
    // 初始化空闲链表 (使用 M_max 作为初始块大小)
    free_list_init(max_arena_bytes);
    
    // 复制 ref_count 以供模拟
    uint8_t *sim_ref = (uint8_t*)malloc(ctx->num_tensors);
    for (uint32_t i = 0; i < ctx->num_tensors; i++) {
        sim_ref[i] = ctx->tensors[i].ref_count;
    }
    
    // 标记模型输出 tensor: 不允许释放 (即使 ref_count=0)
    // 原因: 输出 tensor 没有下游消费者, ref_count 可能为 0,
    // 但它们的值需要在推理结束后被读取
    uint8_t *is_output = (uint8_t*)calloc(ctx->num_tensors, 1);
    for (int i = 0; i < ctx->header.num_outputs; i++) {
        is_output[ctx->output_ids[i]] = 1;
    }
    
    uint32_t peak = 0;
    
    // *** 关键修复：为模型输入tensor预先分配offset ***
    // 原因：模型输入不是任何节点的输出，不会在遍历节点时被分配
    // 如果不预分配，它们的offset默认为0，会与第一个节点输出冲突
    uint16_t num_inputs = ctx->header.num_inputs;
    for (uint16_t i = 0; i < num_inputs; i++) {
        uint16_t inp_id = ctx->input_ids[i];
        SpinnTensor *t = &ctx->tensors[inp_id];
        
        if (t->is_weight) continue;  // 输入不应该是权重
        if (t->size == 0) continue;
        
        t->offset = alloc_offset(t->size);
        if (t->offset == 0xFFFFFFFF) {
            // 分配失败：记录诊断信息
            g_failed_tensor_id = inp_id;
            g_failed_required_size = (t->size + 15) & ~15;
            free(sim_ref);
            free_list_destroy();
            return SPINN_MEM_ERR_NOMEM;
        }
        
        uint32_t aligned = (t->size + 15) & ~15;
        uint32_t end = t->offset + aligned;
        if (end > peak) peak = end;
    }
    
    // 遍历所有节点，模拟执行
    for (uint32_t n = 0; n < ctx->num_nodes; n++) {
        SpinnNode *node = &ctx->nodes[n];
        

        
        // 1. 为输出 Tensor 分配 Offset
        for (int j = 0; j < node->num_outputs; j++) {
            uint16_t out_id = node->output_ids[j];
            SpinnTensor *t = &ctx->tensors[out_id];
            
            if (t->is_weight) continue; // 权重不需要分配
            if (t->size == 0) continue; // 空 Tensor 跳过
            
            t->offset = alloc_offset(t->size);

            if (t->offset == 0xFFFFFFFF) {
                // 分配失败：超过内存上限
                g_failed_tensor_id = out_id;
                g_failed_required_size = (t->size + 15) & ~15;
                free(sim_ref);
                free_list_destroy();
                return SPINN_MEM_ERR_NOMEM;
            }
            
            uint32_t aligned = (t->size + 15) & ~15;
            uint32_t end = t->offset + aligned;
            // 正确的 peak 计算：最大的右边界
            if (end > peak) peak = end;
        }
        
        // 2. 减少输入的引用计数，归零则释放 (模型输出除外)
        for (int j = 0; j < node->num_inputs; j++) {
            uint16_t in_id = node->input_ids[j];
            SpinnTensor *t = &ctx->tensors[in_id];
            
            if (t->is_weight) continue;
            if (t->size == 0) continue;
            if (is_output[in_id]) continue;  /* 模型输出永不释放 */
            
            if (sim_ref[in_id] > 0) {
                sim_ref[in_id]--;
                if (sim_ref[in_id] == 0) {
                    free_offset(t->offset, t->size);
                }
            }
        }
    }
    
    free(sim_ref);
    free(is_output);
    free_list_destroy();
    
    // 保存峰值
    ctx->arena.peak_usage = peak;
    
    // 返回正整数 (峰值)，如果 > INT_MAX 则截断
    return (peak > 0x7FFFFFFF) ? 0x7FFFFFFF : (int)peak;
}

