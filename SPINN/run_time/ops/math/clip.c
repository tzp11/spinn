#include "../kernels.h"
#include "../../../ennf_op_params.h"

/* Clip: out = clamp(in, min, max) */
int op_clip(SpinnTensor **in, int n_in,
            void *params, uint16_t params_size,
            SpinnTensor **out, int n_out) {
    if (n_in < 1 || n_out < 1) return -1;
    
    float *x = (float*)in[0]->data;
    float *y = (float*)out[0]->data;
    uint32_t n = in[0]->elem_count;
    
    float min_val = -3.4028235e+38f;
    float max_val = 3.4028235e+38f;
    
    if (params && params_size >= sizeof(ENNF_ClipParams)) {
        ENNF_ClipParams *p = (ENNF_ClipParams*)params;
        min_val = p->min_val;
        max_val = p->max_val;
    }
    if (n_in >= 2 && in[1]->data && in[1]->elem_count > 0) {
        min_val = ((float*)in[1]->data)[0];
    }
    if (n_in >= 3 && in[2]->data && in[2]->elem_count > 0) {
        max_val = ((float*)in[2]->data)[0];
    }
    
    for (uint32_t i = 0; i < n; i++) {
        float v = x[i];
        if (v < min_val) v = min_val;
        if (v > max_val) v = max_val;
        y[i] = v;
    }
    return 0;
}
