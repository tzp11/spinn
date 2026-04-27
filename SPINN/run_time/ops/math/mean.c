#include "../kernels.h"

/* Mean: 逐元素平均值 (支持多输入) */
int op_mean(SpinnTensor **in, int n_in,
            void *params, uint16_t params_size,
            SpinnTensor **out, int n_out) {
    (void)params; (void)params_size;
    if (n_in < 1 || n_out < 1) return -1;
    
    float *y = (float*)out[0]->data;
    uint32_t n = out[0]->elem_count;
    
    // 初始化为0
    for (uint32_t i = 0; i < n; i++) {
        y[i] = 0;
    }
    
    // 累加所有输入
    for (int idx = 0; idx < n_in; idx++) {
        float *x = (float*)in[idx]->data;
        uint32_t n_x = in[idx]->elem_count;
        for (uint32_t i = 0; i < n; i++) {
            y[i] += x[i % n_x];
        }
    }
    
    // 除以输入数量
    for (uint32_t i = 0; i < n; i++) {
        y[i] /= n_in;
    }
    return 0;
}
