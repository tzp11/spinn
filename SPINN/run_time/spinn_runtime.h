/*
 * spinn_runtime.h - SPINN 推理引擎核心数据结构
 * 
 * 设计原则:
 *   - 静态内存规划 (Offset-based Arena)
 *   - 权重常驻文件句柄
 *   - 三阶段流程: Load -> Plan -> Run
 */

#ifndef __SPINN_RUNTIME_H__
#define __SPINN_RUNTIME_H__

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include "../ennf_def.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * 运行时 Tensor 结构
 * ============================================================ */
typedef struct {
    /* 来自 ENNF TensorMeta */
    uint16_t id;
    uint8_t  dtype;
    uint8_t  ndim;
    uint32_t dims[ENNF_MAX_DIMS];
    uint32_t elem_count;
    uint8_t  is_weight;
    
    /* 运行时扩展 */
    uint8_t  ref_count;         /* 剩余引用计数 (存活代数) */
    uint32_t size;              /* 字节大小 = elem_count * dtype_size */
    uint32_t offset;            /* Arena 内偏移量 (由规划器计算) */
    void    *data;              /* 指向实际数据: arena_base + offset 或 权重指针 */
    void    *packed_data;       /* 用于静态预打包(Offline Pack)的数据 */
} SpinnTensor;

/* ============================================================
 * 运行时 Node 结构
 * ============================================================ */
#define SPINN_MAX_IO 8          /* 单节点最大输入/输出数 */

typedef struct {
    uint16_t op_type;
    uint8_t  num_inputs;
    uint8_t  num_outputs;
    uint16_t input_ids[SPINN_MAX_IO];
    uint16_t output_ids[SPINN_MAX_IO];
    uint16_t params_size;
    void    *params;            /* 指向参数内存 (常驻) */
} SpinnNode;

/* ============================================================
 * 内存 Arena
 * ============================================================ */
typedef struct {
    uint8_t *base;              /* Arena 起始地址 */
    size_t   capacity;          /* Arena 总容量 */
    size_t   peak_usage;        /* 规划时计算的峰值使用量 */
} SpinnArena;

/* ============================================================
 * 运行时上下文 (核心)
 * ============================================================ */
typedef struct {
    /* 模型元信息 */
    ENNF_Header header;
    uint16_t    input_ids[16];      /* 模型输入 Tensor ID 列表 */
    uint16_t    output_ids[16];     /* 模型输出 Tensor ID 列表 */
    
    /* Tensor 池 */
    SpinnTensor *tensors;           /* [num_tensors] */
    uint32_t     num_tensors;
    
    /* Node 池 */
    SpinnNode   *nodes;             /* [num_nodes] */
    uint32_t     num_nodes;
    
    /* 内存管理 */
    SpinnArena   arena;             /* FeatureMap Arena */
    
    /* 权重 I/O */
    FILE        *ennf_fp;           /* 常驻 ENNF 文件句柄 */
    uint32_t     weight_base_offset;/* 权重数据在文件中的起始位置 */
    
    /* 权重缓存表 (可选) */
    void       **weight_cache;      /* [num_weights] - 缓存已加载的权重 */
    uint32_t     num_weights;
    
    /* 标志 */
    uint16_t     flags;
} SpinnContext;

/* ============================================================
 * 公共 API
 * ============================================================ */

/**
 * 阶段 1: 加载模型
 * @param path ENNF 文件路径
 * @return 新分配的 SpinnContext, 失败返回 NULL
 */
SpinnContext* spinn_load(const char *path);

/**
 * 阶段 2: 规划内存
 * - 形状推理 (若需要)
 * - Offset 分配
 * - Arena 分配
 * @param ctx 上下文
 * @param max_arena_bytes Arena 内存上限 (0 表示无限制)
 * @return 0 成功, <0 失败 (SPINN_MEM_ERR_NOMEM 表示超过内存上限)
 */
int spinn_plan(SpinnContext *ctx, uint32_t max_arena_bytes);

/**
 * 阶段 3: 执行推理
 * @param input_data 输入数据指针 (由调用者填充)
 * @param output_data 输出数据指针 (由函数填充)
 * @return 0 成功, <0 失败
 */
int spinn_run(SpinnContext *ctx, void *input_data, void *output_data);

/**
 * 释放资源
 */
void spinn_free(SpinnContext *ctx);

/**
 * 打印 per-op 计时统计 (需要环境变量 SPINN_PROFILE=1 启用)
 */
void spinn_profile_dump(void);

/**
 * 获取权重数据 (Lazy Load)
 * @param ctx 上下文
 * @param tensor_id 权重 Tensor ID
 * @return 权重数据指针, 失败返回 NULL
 */
void* spinn_get_weight(SpinnContext *ctx, uint16_t tensor_id);

#ifdef __cplusplus
}
#endif

#endif /* __SPINN_RUNTIME_H__ */
