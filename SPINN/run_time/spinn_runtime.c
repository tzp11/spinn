/*
 * spinn_runtime.c - SPINN 推理引擎核心实现
 * 
 * 三阶段流程:
 *   1. Load   - 解析 ENNF，填充 SpinnContext
 *   2. Plan   - 内存规划 (Offset 分配)
 *   3. Run    - 按序执行算子
 */

#include "spinn_runtime.h"
#include "spinn_memory_planner.h"
#include "spinn_ops.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================
 * 阶段 1: 加载模型
 * ============================================================ */
SpinnContext* spinn_load(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    
    SpinnContext *ctx = (SpinnContext*)calloc(1, sizeof(SpinnContext));
    if (!ctx) { fclose(fp); return NULL; }
    
    // 读取 Header
    if (fread(&ctx->header, sizeof(ENNF_Header), 1, fp) != 1) {
        goto fail;
    }
    
    // 验证魔数
    if (ctx->header.magic != ENNF_MAGIC) {
        goto fail;
    }
    
    ctx->flags = ctx->header.flags;
    ctx->num_tensors = ctx->header.num_tensors;
    ctx->num_nodes = ctx->header.num_nodes;
    ctx->num_weights = ctx->header.num_weights;
    ctx->weight_base_offset = ctx->header.weight_data_offset;
    
    // 读取图元数据 (输入/输出 ID)
    fseek(fp, ctx->header.graph_meta_offset, SEEK_SET);
    uint16_t num_inputs, num_outputs;
    if (fread(&num_inputs, 2, 1, fp) != 1) goto fail;
    if (fread(ctx->input_ids, 2, num_inputs, fp) != num_inputs) goto fail;
    if (fread(&num_outputs, 2, 1, fp) != 1) goto fail;
    if (fread(ctx->output_ids, 2, num_outputs, fp) != num_outputs) goto fail;
    
    // 分配 Tensor 池
    ctx->tensors = (SpinnTensor*)calloc(ctx->num_tensors, sizeof(SpinnTensor));
    if (!ctx->tensors) goto fail;
    
    // 读取 TensorMeta
    fseek(fp, ctx->header.tensor_meta_offset, SEEK_SET);
    for (uint32_t i = 0; i < ctx->num_tensors; i++) {
        ENNF_TensorMeta meta;
        if (fread(&meta, sizeof(ENNF_TensorMeta), 1, fp) != 1) goto fail;
        
        SpinnTensor *t = &ctx->tensors[meta.tensor_id];
        t->id = meta.tensor_id;
        t->dtype = meta.dtype;
        t->ndim = meta.ndim;
        memcpy(t->dims, meta.dims, sizeof(meta.dims));
        t->elem_count = meta.elem_count;
        t->is_weight = meta.is_weight;
        t->ref_count = meta.ref_count;
        t->size = meta.elem_count * ennf_dtype_size((ENNF_DataType)meta.dtype);
        t->offset = 0;
        t->data = NULL;
    }
    
    // 分配 Node 池
    ctx->nodes = (SpinnNode*)calloc(ctx->num_nodes, sizeof(SpinnNode));
    if (!ctx->nodes) goto fail;
    
    // 读取 NodeTable
    fseek(fp, ctx->header.node_table_offset, SEEK_SET);
    for (uint32_t i = 0; i < ctx->num_nodes; i++) {
        ENNF_NodeBase base;
        if (fread(&base, sizeof(ENNF_NodeBase), 1, fp) != 1) goto fail;
        
        SpinnNode *node = &ctx->nodes[i];
        node->op_type = base.op_type;
        node->num_inputs = base.num_inputs;
        node->num_outputs = base.num_outputs;
        node->params_size = base.params_size;
        
        // 读取输入 ID
        if (node->num_inputs > SPINN_MAX_IO) node->num_inputs = SPINN_MAX_IO;
        if (fread(node->input_ids, 2, node->num_inputs, fp) != node->num_inputs) goto fail;
        
        // 读取输出 ID
        if (node->num_outputs > SPINN_MAX_IO) node->num_outputs = SPINN_MAX_IO;
        if (fread(node->output_ids, 2, node->num_outputs, fp) != node->num_outputs) goto fail;
        
        // 读取参数 (常驻内存)
        if (node->params_size > 0) {
            node->params = malloc(node->params_size);
            if (node->params) {
                if (fread(node->params, 1, node->params_size, fp) != node->params_size) goto fail;
            }
        } else {
            node->params = NULL;
        }
    }
    
    // 分配权重缓存表
    ctx->weight_cache = (void**)calloc(ctx->num_tensors, sizeof(void*));
    
    // 保持文件句柄常驻
    ctx->ennf_fp = fp;
    
    return ctx;
    
fail:
    if (ctx->tensors) free(ctx->tensors);
    if (ctx->nodes) free(ctx->nodes);
    free(ctx);
    fclose(fp);
    return NULL;
}

/* ============================================================
 * 阶段 2: 规划内存
 * ============================================================ */
int spinn_plan(SpinnContext *ctx, uint32_t max_arena_bytes) {
    if (!ctx) return -1;
    
    // TODO: 如果 DYNAMIC_SHAPE 标志开启，先做形状推理
    // if (ctx->flags & ENNF_FLAG_DYNAMIC_SHAPE) {
    //     spinn_shape_inference(ctx);
    // }
    
    // 执行 Offset 规划 (带 M_max 约束)
    int peak = spinn_plan_memory(ctx, max_arena_bytes);
    if (peak < 0) return peak;
    
    // 分配 Arena
    ctx->arena.capacity = (size_t)peak;
    ctx->arena.base = (uint8_t*)calloc(1, peak);
    if (!ctx->arena.base && peak > 0) return -2;
    
    // 为每个 FeatureMap Tensor 设置 data 指针
    for (uint32_t i = 0; i < ctx->num_tensors; i++) {
        SpinnTensor *t = &ctx->tensors[i];
        if (!t->is_weight && t->size > 0) {
            t->data = ctx->arena.base + t->offset;
        }
    }
    
    return 0;
}

/* ============================================================
 * 调试辅助: 打印 Tensor 统计信息
 * ============================================================ */
void print_tensor_stats(SpinnContext *ctx, uint16_t tensor_id, const char *tag) {
    SpinnTensor *t = &ctx->tensors[tensor_id];
    
    // 确保权重已加载
    if (t->is_weight && !t->data) {
         t->data = spinn_get_weight(ctx, tensor_id);
    }
    
    if (!t->data) {
        printf("    %s T%d: (NULL)\n", tag, tensor_id);
        return;
    }
    
    // Check dtype
    // 1=FLOAT, 7=INT64, 6=INT32 (assuming standard ONNX map or ENNF map)
    // In ENNF, usually matches ONNX proto but simplified.
    // Let's assume standard checks:
    
    printf("    %s T%d (%s): shape=[", tag, tensor_id, t->is_weight ? "W" : "A");
    for(int i=0; i<t->ndim; i++) printf("%d%s", t->dims[i], i==t->ndim-1?"":",");
    printf("]");

    uint32_t n = t->elem_count;
    if (n == 0) {
        printf(" (Empty)\n");
        return;
    }

    if (t->dtype == 1) { // Float
        float *data = (float*)t->data;
        float min_v = 1e30f, max_v = -1e30f;
        double sum = 0, sum_sq = 0;
        
        for (uint32_t i = 0; i < n; i++) {
            float v = data[i];
            if (v < min_v) min_v = v;
            if (v > max_v) max_v = v;
            sum += v;
            sum_sq += v * v;
        }
        double mean = sum / n;
        double var = (sum_sq / n) - (mean * mean);
        if (var < 0) var = 0;
        
        printf(" mean=%.6f std=%.6f min=%.6f max=%.6f val=[%.4f %.4f]\n", 
               mean, sqrt(var), min_v, max_v, data[0], n>1?data[1]:0);
               
    } else if (t->dtype == 7) { // INT64
        int64_t *data = (int64_t*)t->data;
        int64_t min_v = data[0], max_v = data[0];
        double sum = 0;
        
        for (uint32_t i = 0; i < n; i++) {
            int64_t v = data[i];
            if (v < min_v) min_v = v;
            if (v > max_v) max_v = v;
            sum += (double)v;
        }
        double mean = sum / n;
        
        printf(" type=INT64 mean=%.6f min=%ld max=%ld val=[%ld %ld]\n", 
               mean, min_v, max_v, data[0], n>1?data[1]:0);
               
    } else if (t->dtype == 6) { // INT32
        int32_t *data = (int32_t*)t->data;
        int32_t min_v = data[0], max_v = data[0];
        double sum = 0;
        
        for (uint32_t i = 0; i < n; i++) {
            int32_t v = data[i];
            if (v < min_v) min_v = v;
            if (v > max_v) max_v = v;
            sum += (double)v;
        }
        double mean = sum / n;
        
        printf(" type=INT32 mean=%.6f min=%d max=%d val=[%d %d]\n", 
               mean, min_v, max_v, data[0], n>1?data[1]:0);
               
    } else {
        printf(" type=%d (Unsupported stats)\n", t->dtype);
    }
}

/* ============================================================
 * 阶段 3: 执行推理
 * ============================================================ */
int spinn_run(SpinnContext *ctx, void *input_data, void *output_data) {
    if (!ctx) return -1;
    
    // 复制输入数据
    SpinnTensor *t_in = &ctx->tensors[ctx->input_ids[0]];
    if (input_data && t_in->data) {
        memcpy(t_in->data, input_data, t_in->size);
    }
    
    // 遍历所有节点执行
    for (uint32_t i = 0; i < ctx->num_nodes; i++) {
        SpinnNode *node = &ctx->nodes[i];
        
        // 准备输入指针
        SpinnTensor *inputs[SPINN_MAX_IO];
        for (int j = 0; j < node->num_inputs; j++) {
            inputs[j] = &ctx->tensors[node->input_ids[j]];
            // 确保权重加载
            if (inputs[j]->is_weight && !inputs[j]->data) {
                inputs[j]->data = spinn_get_weight(ctx, inputs[j]->id);
            }
        }
        
        // 准备输出指针
        SpinnTensor *outputs[SPINN_MAX_IO];
        for (int j = 0; j < node->num_outputs; j++) {
            outputs[j] = &ctx->tensors[node->output_ids[j]];
        }
        
        // 调度算子
        spinn_dispatch_op(node->op_type,
                          inputs, node->num_inputs,
                          node->params, node->params_size,
                          outputs, node->num_outputs);
    }
    
    // 复制输出数据
    SpinnTensor *t_out = &ctx->tensors[ctx->output_ids[0]];
    if (output_data && t_out->data) {
        memcpy(output_data, t_out->data, t_out->size);
    }
    
    return 0;
}

/* ============================================================
 * 权重读取 (Lazy Load)
 * ============================================================ */
/* ============================================================
 * 权重读取 (Lazy Load)
 * ============================================================ */
void* spinn_get_weight(SpinnContext *ctx, uint16_t tensor_id) {
    if (!ctx || tensor_id >= ctx->num_tensors) return NULL;
    
    // 检查缓存
    if (ctx->weight_cache && ctx->weight_cache[tensor_id]) {
        return ctx->weight_cache[tensor_id];
    }
    
    SpinnTensor *t = &ctx->tensors[tensor_id];
    if (!t->is_weight) return NULL;
    
    // 正确的逻辑: 读取 WeightEntry 表
    // WeightEntryTable 的偏移存在 header.reserved[0] 中
    uint32_t entry_table_offset = ctx->header.reserved[0];
    
    // 计算目标 entry 的位置
    uint32_t entry_pos = entry_table_offset + tensor_id * sizeof(ENNF_WeightEntry);
    
    fseek(ctx->ennf_fp, entry_pos, SEEK_SET);
    ENNF_WeightEntry entry;
    if (fread(&entry, sizeof(ENNF_WeightEntry), 1, ctx->ennf_fp) != 1) {
        return NULL;
    }
    
    // 验证 size
    if (entry.data_size == 0) return NULL; // 空权重
    
    // 实际数据位置 = weight_base_offset + entry.data_offset
    uint32_t data_pos = ctx->weight_base_offset + entry.data_offset;
    
    fseek(ctx->ennf_fp, data_pos, SEEK_SET);
    
    void *buf = malloc(entry.data_size);
    if (!buf) return NULL;
    
    if (fread(buf, 1, entry.data_size, ctx->ennf_fp) != entry.data_size) {
        free(buf);
        return NULL;
    }
    
    // 缓存
    if (ctx->weight_cache) {
        ctx->weight_cache[tensor_id] = buf;
    }
    
    return buf;
}

/* ============================================================
 * 释放资源
 * ============================================================ */
void spinn_free(SpinnContext *ctx) {
    if (!ctx) return;
    
    // 释放权重缓存
    if (ctx->weight_cache) {
        for (uint32_t i = 0; i < ctx->num_tensors; i++) {
            if (ctx->weight_cache[i]) free(ctx->weight_cache[i]);
        }
        free(ctx->weight_cache);
    }
    
    // 释放 Arena
    if (ctx->arena.base) free(ctx->arena.base);
    
    // 释放 Node params
    if (ctx->nodes) {
        for (uint32_t i = 0; i < ctx->num_nodes; i++) {
            if (ctx->nodes[i].params) free(ctx->nodes[i].params);
        }
        free(ctx->nodes);
    }
    
    // 释放 Tensors 及 packed_data
    if (ctx->tensors) {
        for (uint32_t i = 0; i < ctx->num_tensors; i++) {
            if (ctx->tensors[i].packed_data) free(ctx->tensors[i].packed_data);
        }
        free(ctx->tensors);
    }
    
    // 关闭文件
    if (ctx->ennf_fp) fclose(ctx->ennf_fp);
    
    free(ctx);
}
