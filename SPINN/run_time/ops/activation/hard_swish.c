#include "../kernels.h"

/* HardSwish: out = x * max(0, min(1, (x + 3) / 6)) */
int op_hard_swish(SpinnTensor **in, int n_in,
                  void *params, uint16_t params_size,
                  SpinnTensor **out, int n_out) {
    (void)params; (void)params_size;
    if (n_in < 1 || n_out < 1) return -1;
    
    float *x = (float*)in[0]->data;
    float *y = (float*)out[0]->data;
    uint32_t n = in[0]->elem_count;
    
    for (uint32_t i = 0; i < n; i++) {
        float v = x[i];
        float h = (v + 3.0f) / 6.0f;
        if (h < 0) h = 0;
        if (h > 1) h = 1;
        y[i] = v * h;
    }
    return 0;
}
