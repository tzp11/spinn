#include "../kernels.h"
#include <math.h>

/* ReduceLogSum: log(sum(x)) */
int op_reduce_log_sum(SpinnTensor **in, int n_in,
                      void *params, uint16_t params_size,
                      SpinnTensor **out, int n_out) {
    if (n_in < 1 || n_out < 1) return -1;
    
    float *X = (float*)in[0]->data;
    float *Y = (float*)out[0]->data;
    
    uint32_t n = in[0]->elem_count;
    double sum = 0;
    for (uint32_t i = 0; i < n; i++) sum += X[i];
    Y[0] = logf(sum);
    return 0;
}
