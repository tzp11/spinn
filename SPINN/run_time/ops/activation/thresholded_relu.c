#include "../kernels.h"

/* ThresholdedRelu: y = x if x > alpha else 0 */
int op_thresholded_relu(SpinnTensor **in, int n_in,
                        void *params, uint16_t params_size,
                        SpinnTensor **out, int n_out) {
    if (n_in < 1 || n_out < 1) return -1;
    
    float *X = (float*)in[0]->data;
    float *Y = (float*)out[0]->data;
    
    float alpha = 1.0f; // Default
    // params parsing omitted
    
    uint32_t n = in[0]->elem_count;
    for (uint32_t i = 0; i < n; i++) {
        Y[i] = (X[i] > alpha) ? X[i] : 0.0f;
    }
    return 0;
}
