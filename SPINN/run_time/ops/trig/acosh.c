#include "../kernels.h"
#include <math.h>
int op_acosh(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out) {
    (void)params; (void)params_size;
    if (n_in < 1 || n_out < 1) return -1;
    float *x = (float*)in[0]->data; float *y = (float*)out[0]->data;
    for (uint32_t i = 0; i < in[0]->elem_count; i++) y[i] = acoshf(x[i]);
    return 0;
}
