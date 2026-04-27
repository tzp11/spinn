#include "../kernels.h"
#include <math.h>

/* LayerNormalization */
int op_layer_norm(SpinnTensor **in, int n_in,
                  void *params, uint16_t params_size,
                  SpinnTensor **out, int n_out) {
    (void)params; (void)params_size;
    if (n_in < 1 || n_out < 1) return -1;
    
    float *X = (float*)in[0]->data;
    float *Y = (float*)out[0]->data;
    float *scale = (n_in > 1 && in[1]->data) ? (float*)in[1]->data : NULL;
    float *bias = (n_in > 2 && in[2]->data) ? (float*)in[2]->data : NULL;
    float eps = 1e-5f;
    
    uint32_t total = in[0]->elem_count;
    int last_dim = in[0]->dims[in[0]->ndim - 1];
    int num_instances = total / last_dim;
    
    for (int i = 0; i < num_instances; i++) {
        float *ptr = X + i * last_dim;
        float *out_ptr = Y + i * last_dim;
        
        float mean = 0, var = 0;
        for (int j = 0; j < last_dim; j++) mean += ptr[j];
        mean /= last_dim;
        for (int j = 0; j < last_dim; j++) var += (ptr[j] - mean) * (ptr[j] - mean);
        var /= last_dim;
        
        float inv_std = 1.0f / sqrtf(var + eps);
        
        for (int j = 0; j < last_dim; j++) {
            float s = scale ? scale[j] : 1.0f;
            float b = bias ? bias[j] : 0.0f;
            out_ptr[j] = (ptr[j] - mean) * inv_std * s + b;
        }
    }
    return 0;
}
