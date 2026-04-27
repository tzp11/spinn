#include "../kernels.h"
#include <stdint.h>

/* BitShift: out = in[0] << in[1] (LEFT) or in[0] >> in[1] (RIGHT) */
/* 假设 params[0]: 0=LEFT, 1=RIGHT */
int op_bitshift(SpinnTensor **in, int n_in,
                void *params, uint16_t params_size,
                SpinnTensor **out, int n_out) {
    if (n_in < 2 || n_out < 1) return -1;
    
    uint8_t direction = 0; // 0=LEFT, 1=RIGHT
    if (params && params_size >= 1) {
        direction = ((uint8_t*)params)[0];
    }
    
    // 支持多种整型，这里简化为 int32 (ONNX 常见类型)
    int32_t *a = (int32_t*)in[0]->data;
    int32_t *b = (int32_t*)in[1]->data;
    int32_t *y = (int32_t*)out[0]->data;
    
    uint32_t n = out[0]->elem_count;
    uint32_t n_a = in[0]->elem_count;
    uint32_t n_b = in[1]->elem_count;
    
    if (direction == 0) { // LEFT
        for (uint32_t i = 0; i < n; i++) {
            y[i] = a[i % n_a] << b[i % n_b];
        }
    } else { // RIGHT
        for (uint32_t i = 0; i < n; i++) {
            y[i] = a[i % n_a] >> b[i % n_b];
        }
    }
    return 0;
}
