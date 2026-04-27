#include "../kernels.h"
#include "../../../ennf_op_params.h"

/* HardSigmoid: out = max(0, min(1, alpha*x + beta)) */
int op_hard_sigmoid(SpinnTensor **in, int n_in,
                    void *params, uint16_t params_size,
                    SpinnTensor **out, int n_out) {
    if (n_in < 1 || n_out < 1) return -1;
    
    float alpha = 0.2f;
    float beta = 0.5f;
    if (params && params_size >= sizeof(ENNF_HardSigmoidParams)) {
        ENNF_HardSigmoidParams *p = (ENNF_HardSigmoidParams*)params;
        alpha = p->alpha;
        beta = p->beta;
    }
    
    float *x = (float*)in[0]->data;
    float *y = (float*)out[0]->data;
    uint32_t n = in[0]->elem_count;
    
    for (uint32_t i = 0; i < n; i++) {
        float v = alpha * x[i] + beta;
        if (v < 0) v = 0;
        if (v > 1) v = 1;
        y[i] = v;
    }
    return 0;
}
