#include "../kernels.h"
#include "../../../ennf_op_params.h"
#include <math.h>

/* ReduceL2: L2范数归约 */
int op_reduce_l2(SpinnTensor **in, int n_in,
                 void *params, uint16_t params_size,
                 SpinnTensor **out, int n_out) {
    if (n_in < 1 || n_out < 1) return -1;
    
    float *X = (float*)in[0]->data;
    float *Y = (float*)out[0]->data;
    
    uint32_t n = in[0]->elem_count;
    double sum_sq = 0;
    for (uint32_t i = 0; i < n; i++) {
        sum_sq += X[i] * X[i];
    }
    Y[0] = sqrt(sum_sq);
    return 0;
}
