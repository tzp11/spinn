/*
 * main.c - SPINN Runtime 测试入口 (带精确推理计时)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "spinn_runtime.h"

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <model.ennf> [num_runs] [--max-arena=SIZE]\n", argv[0]);
        printf("  SIZE examples: 4M, 16M, 64M, 256M (0 = unlimited)\n");
        return 1;
    }
    
    const char *model_path = argv[1];
    int num_runs = 1;
    uint32_t max_arena = 0;  // 0 = 无限制
    
    // 解析参数
    for (int i = 2; i < argc; i++) {
        if (strncmp(argv[i], "--max-arena=", 12) == 0) {
            const char *val = argv[i] + 12;
            char *end;
            max_arena = (uint32_t)strtoul(val, &end, 10);
            if (*end == 'K' || *end == 'k') max_arena *= 1024;
            else if (*end == 'M' || *end == 'm') max_arena *= 1024 * 1024;
            else if (*end == 'G' || *end == 'g') max_arena *= 1024 * 1024 * 1024;
        } else {
            num_runs = atoi(argv[i]);
        }
    }
    if (num_runs < 1) num_runs = 1;
    int warmup = (num_runs > 1) ? 3 : 0;
    
    /* Stage 1: Load */
    double t0 = get_time_ms();
    SpinnContext *ctx = spinn_load(model_path);
    if (!ctx) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }
    double t_load = get_time_ms() - t0;
    
    /* Stage 2: Plan (带 M_max 约束) */
    t0 = get_time_ms();
    int ret = spinn_plan(ctx, max_arena);
    if (ret < 0) {
        if (ret == -2) {
            fprintf(stderr, "Memory planning failed: Arena peak exceeds limit (%u bytes)\n", max_arena);
        } else {
            fprintf(stderr, "Failed to plan memory: %d\n", ret);
        }
        spinn_free(ctx);
        return 1;
    }
    double t_plan = get_time_ms() - t0;
    
    /* 准备输入 */
    SpinnTensor *t_in = &ctx->tensors[ctx->input_ids[0]];
    float *input_data = (float*)malloc(t_in->size);
    for (uint32_t i = 0; i < t_in->elem_count; i++) {
        input_data[i] = (float)i / 1000.0f;
    }
    
    SpinnTensor *t_out = &ctx->tensors[ctx->output_ids[0]];
    float *output_data = (float*)malloc(t_out->size);
    
    if (num_runs > 1) {
        fprintf(stderr, "Load: %.1fms, Plan: %.1fms\n", t_load, t_plan);
        fprintf(stderr, "Warmup: %d, Runs: %d\n", warmup, num_runs);
    }
    
    /* Warmup */
    for (int i = 0; i < warmup; i++) {
        spinn_run(ctx, input_data, output_data);
    }
    
    /* Benchmark */
    double best = 1e9, total = 0;
    for (int i = 0; i < num_runs; i++) {
        t0 = get_time_ms();
        ret = spinn_run(ctx, input_data, output_data);
        double elapsed = get_time_ms() - t0;
        
        if (ret < 0) {
            fprintf(stderr, "Inference failed: %d\n", ret);
            break;
        }
        
        if (elapsed < best) best = elapsed;
        total += elapsed;
        
        if (num_runs > 1) {
            fprintf(stderr, "  Run %d: %.1fms\n", i + 1, elapsed);
        }
    }
    
    if (num_runs > 1) {
        fprintf(stderr, "Best: %.1fms, Avg: %.1fms\n", best, total / num_runs);
    }
    
    /* 输出结果 (stdout) */
    printf("SPINN Result: ");
    int n = t_out->elem_count;
    for (int i = 0; i < n; i++) {
        printf("%.6f ", output_data[i]);
    }
    printf("\n");
    
    /* 二进制输出 dump (用于与 ONNX Runtime 对比) */
    const char *dump_env = getenv("SPINN_DUMP");
    if (dump_env && dump_env[0] != '0') {
        FILE *f = fopen(dump_env, "wb");
        if (f) {
            /* dump 所有输出 tensor */
            for (int oi = 0; oi < ctx->header.num_outputs; oi++) {
                SpinnTensor *to = &ctx->tensors[ctx->output_ids[oi]];
                if (to->data) fwrite(to->data, sizeof(float), to->elem_count, f);
            }
            fclose(f);
            fprintf(stderr, "Dumped %d outputs to %s\n", ctx->header.num_outputs, dump_env);
        }
    }
    
    free(input_data);
    free(output_data);
    spinn_profile_dump();
    spinn_free(ctx);
    return 0;
}
