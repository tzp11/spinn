#include "../kernels.h"
#include "../../../ennf_op_params.h"
#include <math.h>

/* BatchNormalization */
int op_batch_norm(SpinnTensor **in, int n_in,
                  void *params, uint16_t params_size,
                  SpinnTensor **out, int n_out) {
    if (n_in < 5 || n_out < 1) return -1;
    
    float *X = (float*)in[0]->data;
    float *scale = (float*)in[1]->data;
    float *bias = (float*)in[2]->data;
    float *running_mean = (float*)in[3]->data;
    float *running_var = (float*)in[4]->data;
    float *Y = (float*)out[0]->data;
    
    float eps = 1e-5f;
    if (params && params_size >= sizeof(ENNF_BatchNormParams)) {
        ENNF_BatchNormParams *p = (ENNF_BatchNormParams*)params;
        eps = p->epsilon;
    }
    
    int N = in[0]->dims[0];
    int C = in[0]->dims[1];
    int spatial = 1;
    for (int d = 2; d < in[0]->ndim; d++) spatial *= in[0]->dims[d];
    
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            float mean = running_mean[c];
            float var = running_var[c];
            float s = scale[c];
            float b = bias[c];
            float inv_std = 1.0f / sqrtf(var + eps);
            
            float *x_ptr = X + (n * C + c) * spatial;
            float *y_ptr = Y + (n * C + c) * spatial;
            
            for (int i = 0; i < spatial; i++) {
                y_ptr[i] = s * (x_ptr[i] - mean) * inv_std + b;
            }
        }
    }
    return 0;
}
