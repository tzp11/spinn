#include "../kernels.h"
#include "../../../ennf_op_params.h"

/* Shrink: y = x + bias if x < -lambd; y = x - bias if x > lambd; else 0 */
int op_shrink(SpinnTensor **in, int n_in,
              void *params, uint16_t params_size,
              SpinnTensor **out, int n_out) {
    if (n_in < 1 || n_out < 1) return -1;
    
    float *X = (float*)in[0]->data;
    float *Y = (float*)out[0]->data;
    
    float bias = 0.0f;
    float lambd = 0.5f;
    
    if (params && params_size >= sizeof(ENNF_ShrinkParams)) {
        ENNF_ShrinkParams *p = (ENNF_ShrinkParams*)params;
        bias = p->bias;
        lambd = p->lambd;
    }
    
    uint32_t n = in[0]->elem_count;
    for (uint32_t i = 0; i < n; i++) {
        float x = X[i];
        if (x < -lambd) {
            Y[i] = x + bias;
        } else if (x > lambd) {
            Y[i] = x - bias;
        } else {
            Y[i] = 0.0f;
        }
    }
    return 0;
}
