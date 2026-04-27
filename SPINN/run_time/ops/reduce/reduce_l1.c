#include "../kernels.h"
#include "../../../ennf_op_params.h"
#include <math.h>

/* ReduceL1: L1范数归约 */
int op_reduce_l1(SpinnTensor **in, int n_in,
                 void *params, uint16_t params_size,
                 SpinnTensor **out, int n_out) {
    if (n_in < 1 || n_out < 1) return -1;
    
    float *X = (float*)in[0]->data;
    float *Y = (float*)out[0]->data;
    
    int axes[8] = {-1}; // Default: all axes
    int num_axes = 1;
    int keepdims = 1;
    
    if (params && params_size >= sizeof(ENNF_ReduceParams)) {
        ENNF_ReduceParams *p = (ENNF_ReduceParams*)params;
        num_axes = p->num_axes;
        for (int i = 0; i < num_axes; i++) axes[i] = p->axes[i];
        keepdims = p->keepdims;
    }
    (void)axes; (void)keepdims; (void)num_axes;
    
    // 简化实现：归约所有维度
    uint32_t n = in[0]->elem_count;
    double sum = 0;
    for (uint32_t i = 0; i < n; i++) {
        sum += fabs(X[i]);
    }
    Y[0] = sum;
    return 0;
}
