#include "../kernels.h"

/* Shape: 返回张量的形状 */
int op_shape(SpinnTensor **in, int n_in,
             void *params, uint16_t params_size,
             SpinnTensor **out, int n_out) {
    (void)params; (void)params_size;
    if (n_in < 1 || n_out < 1) return -1;
    
    int64_t *Y = (int64_t*)out[0]->data;
    int ndim = in[0]->ndim;
    
    for (int i = 0; i < ndim; i++) {
        Y[i] = in[0]->dims[i];
    }
    return 0;
}
