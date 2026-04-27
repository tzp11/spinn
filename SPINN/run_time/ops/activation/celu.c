#include "../kernels.h"
#include "../../../ennf_op_params.h"
#include <math.h>

/* Celu: out = max(0,x) + min(0, alpha*(exp(x/alpha)-1)) */
int op_celu(SpinnTensor **in, int n_in,
            void *params, uint16_t params_size,
            SpinnTensor **out, int n_out) {
    if (n_in < 1 || n_out < 1) return -1;
    
    float alpha = 1.0f;
    if (params && params_size >= sizeof(ENNF_EluParams)) {
        ENNF_EluParams *p = (ENNF_EluParams*)params;
        alpha = p->alpha;
    }
    
    float *x = (float*)in[0]->data;
    float *y = (float*)out[0]->data;
    uint32_t n = in[0]->elem_count;
    
    for (uint32_t i = 0; i < n; i++) {
        float v = x[i];
        float pos = (v > 0) ? v : 0;
        float neg = (v < 0) ? alpha * (expf(v / alpha) - 1.0f) : 0;
        y[i] = pos + neg;
    }
    return 0;
}
