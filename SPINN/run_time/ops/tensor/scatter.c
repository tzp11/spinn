#include "../kernels.h"

/* Scatter: 根据索引将updates写入data (简化版) */
int op_scatter(SpinnTensor **in, int n_in,
               void *params, uint16_t params_size,
               SpinnTensor **out, int n_out) {
    if (n_in < 3 || n_out < 1) return -1;
    
    // in[0]: data, in[1]: indices, in[2]: updates
    float *data = (float*)in[0]->data;
    int64_t *indices = (int64_t*)in[1]->data;
    float *updates = (float*)in[2]->data;
    float *Y = (float*)out[0]->data;
    
    // 复制data到输出
    uint32_t n = in[0]->elem_count;
    for (uint32_t i = 0; i < n; i++) {
        Y[i] = data[i];
    }
    
    // 简化：假设1D scatter
    uint32_t num_updates = in[1]->elem_count;
    for (uint32_t i = 0; i < num_updates; i++) {
        int64_t idx = indices[i];
        if (idx < 0) idx += n;
        if (idx >= 0 && idx < (int64_t)n) {
            Y[idx] = updates[i];
        }
    }
    return 0;
}
