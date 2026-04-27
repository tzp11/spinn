#include "../kernels.h"

/* And: out = in[0] && in[1] (逻辑与) */
int op_and(SpinnTensor **in, int n_in,
           void *params, uint16_t params_size,
           SpinnTensor **out, int n_out) {
    (void)params; (void)params_size;
    if (n_in < 2 || n_out < 1) return -1;
    
    uint8_t *a = (uint8_t*)in[0]->data;
    uint8_t *b = (uint8_t*)in[1]->data;
    uint8_t *y = (uint8_t*)out[0]->data;
    
    uint32_t n = out[0]->elem_count;
    uint32_t n_a = in[0]->elem_count;
    uint32_t n_b = in[1]->elem_count;
    
    for (uint32_t i = 0; i < n; i++) {
        y[i] = (a[i % n_a] && b[i % n_b]) ? 1 : 0;
    }
    return 0;
}
