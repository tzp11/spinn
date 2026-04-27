#include "../kernels.h"
#include <math.h>

/* InstanceNormalization */
int op_instance_norm(SpinnTensor **in, int n_in,
                     void *params, uint16_t params_size,
                     SpinnTensor **out, int n_out) {
    (void)params; (void)params_size;
    if (n_in < 1 || n_out < 1) return -1;
    
    float *X = (float*)in[0]->data;
    float *Y = (float*)out[0]->data;
    float *scale = (n_in > 1 && in[1]->data) ? (float*)in[1]->data : NULL;
    float *bias = (n_in > 2 && in[2]->data) ? (float*)in[2]->data : NULL;
    float eps = 1e-5f;
    
    int N = in[0]->dims[0];
    int C = in[0]->dims[1];
    int H = (in[0]->ndim >= 3) ? in[0]->dims[2] : 1;
    int W = (in[0]->ndim >= 4) ? in[0]->dims[3] : 1;
    int spatial = H * W;
    
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            float *ptr = X + (n * C + c) * spatial;
            float *out_ptr = Y + (n * C + c) * spatial;
            
            float mean = 0, var = 0;
            for (int i = 0; i < spatial; i++) mean += ptr[i];
            mean /= spatial;
            for (int i = 0; i < spatial; i++) var += (ptr[i] - mean) * (ptr[i] - mean);
            var /= spatial;
            
            float inv_std = 1.0f / sqrtf(var + eps);
            float s = scale ? scale[c] : 1.0f;
            float b = bias ? bias[c] : 0.0f;
            
            for (int i = 0; i < spatial; i++) {
                out_ptr[i] = (ptr[i] - mean) * inv_std * s + b;
            }
        }
    }
    return 0;
}
