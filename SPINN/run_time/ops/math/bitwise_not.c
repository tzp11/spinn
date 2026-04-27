#include "../kernels.h"

/* BitwiseNot: out = ~in */
int op_bitwise_not(SpinnTensor **in, int n_in,
                   void *params, uint16_t params_size,
                   SpinnTensor **out, int n_out) {
    (void)params; (void)params_size;
    if (n_in < 1 || n_out < 1) return -1;
    
    int32_t *x = (int32_t*)in[0]->data;
    int32_t *y = (int32_t*)out[0]->data;
    uint32_t n = in[0]->elem_count;
    
    for (uint32_t i = 0; i < n; i++) {
        y[i] = ~x[i];
    }
    return 0;
}
