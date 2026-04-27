#include "../kernels.h"

/* BitwiseAnd: out = in[0] & in[1] */
int op_bitwise_and(SpinnTensor **in, int n_in,
                   void *params, uint16_t params_size,
                   SpinnTensor **out, int n_out) {
    (void)params; (void)params_size;
    if (n_in < 2 || n_out < 1) return -1;
    
    int32_t *a = (int32_t*)in[0]->data;
    int32_t *b = (int32_t*)in[1]->data;
    int32_t *y = (int32_t*)out[0]->data;
    
    uint32_t n = out[0]->elem_count;
    uint32_t n_a = in[0]->elem_count;
    uint32_t n_b = in[1]->elem_count;
    
    for (uint32_t i = 0; i < n; i++) {
        y[i] = a[i % n_a] & b[i % n_b];
    }
    return 0;
}
