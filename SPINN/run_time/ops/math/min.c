#include "../kernels.h"

/* Min: 逐元素取最小值 */
int op_min(SpinnTensor **in, int n_in,
           void *params, uint16_t params_size,
           SpinnTensor **out, int n_out) {
    (void)params; (void)params_size;
    if (n_in < 1 || n_out < 1) return -1;
    
    float *y = (float*)out[0]->data;
    uint32_t n = out[0]->elem_count;
    
    // 初始化为第一个输入
    float *first = (float*)in[0]->data;
    for (uint32_t i = 0; i < n; i++) {
        y[i] = first[i % in[0]->elem_count];
    }
    
    // 与其他输入比较
    for (int idx = 1; idx < n_in; idx++) {
        float *x = (float*)in[idx]->data;
        uint32_t n_x = in[idx]->elem_count;
        for (uint32_t i = 0; i < n; i++) {
            float val = x[i % n_x];
            if (val < y[i]) y[i] = val;
        }
    }
    return 0;
}
