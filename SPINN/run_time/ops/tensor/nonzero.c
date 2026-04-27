#include "../kernels.h"

/* NonZero: 返回非零元素的索引 */
int op_nonzero(SpinnTensor **in, int n_in,
               void *params, uint16_t params_size,
               SpinnTensor **out, int n_out) {
    (void)params; (void)params_size;
    if (n_in < 1 || n_out < 1) return -1;
    
    // NonZero的输出形状是动态的，需要先统计非零元素数量
    // 在静态runtime中较难处理，这里提供简化版本
    
    float *X = (float*)in[0]->data;
    int64_t *Y = (int64_t*)out[0]->data;
    
    uint32_t n = in[0]->elem_count;
    int ndim = in[0]->ndim;
    
    // 计算strides
    uint32_t strides[8];
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * in[0]->dims[i + 1];
    }
    
    uint32_t nonzero_count = 0;
    for (uint32_t idx = 0; idx < n; idx++) {
        if (X[idx] != 0.0f) {
            // 计算多维索引
            uint32_t tmp = idx;
            for (int d = 0; d < ndim; d++) {
                Y[d * n + nonzero_count] = tmp / strides[d];
                tmp = tmp % strides[d];
            }
            nonzero_count++;
        }
    }
    
    return 0;
}
