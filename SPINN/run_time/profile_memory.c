/*
 * profile_memory.c - 内存分析工具
 * 
 * 独立编译，通过 LD_PRELOAD 或宏替换跟踪 runtime 的内存行为。
 * 由于直接修改 runtime 源码侵入性太强，这里采用另一种方式：
 * 加载模型后，直接计算各部分的真实内存占用。
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "spinn_runtime.h"
#include "../ennf_def.h"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.ennf>\n", argv[0]);
        return 1;
    }

    const char *path = argv[1];

    /* ============================================================
     * Phase 1: 直接读取 ENNF 头，分析元数据
     * ============================================================ */
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", path); return 1; }

    ENNF_Header hdr;
    fread(&hdr, sizeof(ENNF_Header), 1, fp);

    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fclose(fp);

    printf("========================================\n");
    printf("模型: %s\n", path);
    printf("文件大小: %ld bytes (%.2f MB)\n", file_size, file_size/1048576.0);
    printf("========================================\n\n");

    printf("--- ENNF 模型概况 ---\n");
    printf("  节点数:   %u\n", hdr.num_nodes);
    printf("  张量数:   %u\n", hdr.num_tensors);
    printf("  权重数:   %u\n", hdr.num_weights);
    printf("  权重字节: %u (%.2f MB)\n", hdr.total_weight_bytes, hdr.total_weight_bytes/1048576.0);
    printf("\n");

    /* ============================================================
     * Phase 2: 加载并规划，统计真实内存
     * ============================================================ */
    SpinnContext *ctx = spinn_load(path);
    if (!ctx) { fprintf(stderr, "Load failed\n"); return 1; }

    int peak = spinn_plan(ctx);
    if (peak < 0) { fprintf(stderr, "Plan failed\n"); spinn_free(ctx); return 1; }

    /* 计算各部分真实内存 */
    size_t mem_context = sizeof(SpinnContext);
    size_t mem_tensors = ctx->num_tensors * sizeof(SpinnTensor);
    size_t mem_nodes = ctx->num_nodes * sizeof(SpinnNode);
    size_t mem_params = 0;
    for (uint32_t i = 0; i < ctx->num_nodes; i++) {
        mem_params += ctx->nodes[i].params_size;
    }
    size_t mem_weight_cache_table = ctx->num_tensors * sizeof(void*);
    size_t mem_arena = ctx->arena.capacity;

    size_t mem_runtime_infra = mem_context + mem_tensors + mem_nodes + mem_params + mem_weight_cache_table;

    /* 统计权重缓存（需要先执行一次推理来触发 lazy load） */
    SpinnTensor *t_in = &ctx->tensors[ctx->input_ids[0]];
    float *input_data = (float*)calloc(1, t_in->size);
    SpinnTensor *t_out = &ctx->tensors[ctx->output_ids[0]];
    float *output_data = (float*)calloc(1, t_out->size);
    spinn_run(ctx, input_data, output_data);

    size_t mem_weight_data = 0;
    uint32_t weight_loaded = 0;
    for (uint32_t i = 0; i < ctx->num_tensors; i++) {
        if (ctx->weight_cache && ctx->weight_cache[i]) {
            mem_weight_data += ctx->tensors[i].size;
            weight_loaded++;
        }
    }

    size_t mem_total = mem_runtime_infra + mem_arena + mem_weight_data;

    printf("--- 运行时真实内存占用 (推理一次后) ---\n");
    printf("\n");
    printf("  [运行时基础设施]\n");
    printf("    SpinnContext:        %8zu bytes\n", mem_context);
    printf("    Tensor 池:          %8zu bytes (%u 个 × %zu bytes)\n",
           mem_tensors, ctx->num_tensors, sizeof(SpinnTensor));
    printf("    Node 池:            %8zu bytes (%u 个 × %zu bytes)\n",
           mem_nodes, ctx->num_nodes, sizeof(SpinnNode));
    printf("    Node params:        %8zu bytes\n", mem_params);
    printf("    weight_cache 表:    %8zu bytes\n", mem_weight_cache_table);
    printf("    小计:               %8zu bytes (%.2f KB)\n", mem_runtime_infra, mem_runtime_infra/1024.0);
    printf("\n");
    printf("  [Arena (FeatureMap)]\n");
    printf("    Arena 峰值:         %8zu bytes (%.2f KB, %.2f MB)\n",
           mem_arena, mem_arena/1024.0, mem_arena/1048576.0);
    printf("\n");
    printf("  [权重数据 (Lazy Load 后缓存)]\n");
    printf("    已加载权重:         %u / %u 个\n", weight_loaded, hdr.num_weights);
    printf("    权重缓存:           %8zu bytes (%.2f MB)\n",
           mem_weight_data, mem_weight_data/1048576.0);
    printf("\n");

    /* 计算所有 FeatureMap 张量的总大小 vs Arena Peak（复用率） */
    size_t total_featuremap_size = 0;
    uint32_t featuremap_count = 0;
    for (uint32_t i = 0; i < ctx->num_tensors; i++) {
        SpinnTensor *t = &ctx->tensors[i];
        if (!t->is_weight && t->size > 0) {
            total_featuremap_size += t->size;
            featuremap_count++;
        }
    }

    printf("  [内存规划效率]\n");
    printf("    FeatureMap 张量数:  %u\n", featuremap_count);
    printf("    所有 FeatureMap 总大小 (Σsize): %zu bytes (%.2f MB)\n",
           total_featuremap_size, total_featuremap_size/1048576.0);
    printf("    Arena 峰值 (Peak):              %zu bytes (%.2f MB)\n",
           mem_arena, mem_arena/1048576.0);
    if (total_featuremap_size > 0) {
        printf("    内存复用率: Peak/Σsize = %.1f%% (节省 %.1f%%)\n",
               (double)mem_arena / total_featuremap_size * 100.0,
               (1.0 - (double)mem_arena / total_featuremap_size) * 100.0);
    }
    printf("\n");

    printf("  ====================================\n");
    printf("  总运行时内存:         %8zu bytes (%.2f MB)\n",
           mem_total, mem_total/1048576.0);
    printf("    其中 运行时基础设施: %.2f%%\n", (double)mem_runtime_infra/mem_total*100);
    printf("    其中 Arena:         %.2f%%\n", (double)mem_arena/mem_total*100);
    printf("    其中 权重缓存:      %.2f%%\n", (double)mem_weight_data/mem_total*100);
    printf("  ====================================\n");
    printf("\n");

    /* 朴素分配对比 */
    size_t naive_mem = total_featuremap_size + mem_weight_data + mem_runtime_infra;
    printf("  [对比: 朴素分配 (每个张量独立 malloc, 不复用)]\n");
    printf("    朴素总内存: %zu bytes (%.2f MB)\n", naive_mem, naive_mem/1048576.0);
    printf("    本方案总内存: %zu bytes (%.2f MB)\n", mem_total, mem_total/1048576.0);
    if (naive_mem > 0) {
        printf("    节省: %.2f MB (%.1f%%)\n",
               (naive_mem - mem_total)/1048576.0,
               (1.0 - (double)mem_total/naive_mem)*100);
    }

    free(input_data);
    free(output_data);
    spinn_free(ctx);
    return 0;
}
