#include "../kernels.h"
#include <math.h>

/* IsNaN: 检查元素是否为 NaN */
int op_is_nan(SpinnTensor **in, int n_in,
              void *params, uint16_t params_size,
              SpinnTensor **out, int n_out) {
    if (n_in < 1 || n_out < 1) return -1;
    
    float *X = (float*)in[0]->data;
    uint8_t *Y = (uint8_t*)out[0]->data; // Bool
    uint32_t n = in[0]->elem_count;
    
    for (uint32_t i = 0; i < n; i++) {
        Y[i] = isnan(X[i]) ? 1 : 0;
    }
    return 0;
}
