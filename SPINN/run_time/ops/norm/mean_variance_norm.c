#include "../kernels.h"
#include <math.h>

/* MeanVarianceNormalization */
int op_mean_variance_norm(SpinnTensor **in, int n_in,
                          void *params, uint16_t params_size,
                          SpinnTensor **out, int n_out) {
    if (n_in < 1 || n_out < 1) return -1;
    
    float *X = (float*)in[0]->data;
    float *Y = (float*)out[0]->data;
    
    // 简化实现：假设归一化所有维度
    uint32_t n = in[0]->elem_count;
    
    // 计算均值
    double sum = 0;
    for (uint32_t i = 0; i < n; i++) sum += X[i];
    double mean = sum / n;
    
    // 计算方差
    double var_sum = 0;
    for (uint32_t i = 0; i < n; i++) {
        double diff = X[i] - mean;
        var_sum += diff * diff;
    }
    double variance = var_sum / n;
    double std_dev = sqrt(variance + 1e-9); // 加小值避免除零
    
    // 归一化
    for (uint32_t i = 0; i < n; i++) {
        Y[i] = (X[i] - mean) / std_dev;
    }
    return 0;
}
