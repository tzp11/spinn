#include "../kernels.h"
#include <math.h>

/* ReduceLogSumExp: log(sum(exp(x))) */
int op_reduce_log_sum_exp(SpinnTensor **in, int n_in,
                          void *params, uint16_t params_size,
                          SpinnTensor **out, int n_out) {
    if (n_in < 1 || n_out < 1) return -1;
    
    float *X = (float*)in[0]->data;
    float *Y = (float*)out[0]->data;
    
    uint32_t n = in[0]->elem_count;
    
    // 数值稳定性：先找最大值
    float max_val = X[0];
    for (uint32_t i = 1; i < n; i++) {
        if (X[i] > max_val) max_val = X[i];
    }
    
    double sum_exp = 0;
    for (uint32_t i = 0; i < n; i++) {
        sum_exp += expf(X[i] - max_val);
    }
    Y[0] = logf(sum_exp) + max_val;
    return 0;
}
