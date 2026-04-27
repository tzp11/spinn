#include "../kernels.h"

/* Not: 逻辑非 */
int op_not(SpinnTensor **in, int n_in,
           void *params, uint16_t params_size,
           SpinnTensor **out, int n_out) {
    (void)params; (void)params_size;
    if (n_in < 1 || n_out < 1) return -1;
    
    uint8_t *x = (uint8_t*)in[0]->data;
    uint8_t *y = (uint8_t*)out[0]->data;
    uint32_t n = out[0]->elem_count;
    
    for (uint32_t i = 0; i < n; i++) {
        y[i] = x[i] ? 0 : 1;
    }
    return 0;
}
