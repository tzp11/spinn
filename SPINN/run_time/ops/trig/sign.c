#include "../kernels.h"
int op_sign(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out) {
    (void)params; (void)params_size;
    if (n_in < 1 || n_out < 1) return -1;
    float *x = (float*)in[0]->data; float *y = (float*)out[0]->data;
    for (uint32_t i = 0; i < in[0]->elem_count; i++) {
        if (x[i] > 0) y[i] = 1.0f;
        else if (x[i] < 0) y[i] = -1.0f;
        else y[i] = 0.0f;
    }
    return 0;
}
