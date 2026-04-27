#include "../kernels.h"

/* Range: 生成数值范围序列 */
int op_range(SpinnTensor **in, int n_in,
             void *params, uint16_t params_size,
             SpinnTensor **out, int n_out) {
    if (n_in < 3 || n_out < 1) return -1;
    
    // in[0]: start, in[1]: limit, in[2]: delta
    float start = ((float*)in[0]->data)[0];
    float limit = ((float*)in[1]->data)[0];
    float delta = ((float*)in[2]->data)[0];
    
    float *Y = (float*)out[0]->data;
    uint32_t n = out[0]->elem_count;
    
    float val = start;
    for (uint32_t i = 0; i < n; i++) {
        Y[i] = val;
        val += delta;
        if ((delta > 0 && val >= limit) || (delta < 0 && val <= limit)) break;
    }
    return 0;
}
