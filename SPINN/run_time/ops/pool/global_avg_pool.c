#include "../kernels.h"

/* GlobalAveragePool */
int op_global_avg_pool(SpinnTensor **in, int n_in,
                       void *params, uint16_t params_size,
                       SpinnTensor **out, int n_out) {
    (void)params; (void)params_size;
    if (n_in < 1 || n_out < 1) return -1;
    
    float *X = (float*)in[0]->data;
    float *Y = (float*)out[0]->data;
    
    int N = in[0]->dims[0];
    int C = in[0]->dims[1];
    int H = in[0]->dims[2];
    int W_dim = in[0]->dims[3];
    int spatial = H * W_dim;
    
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            float sum = 0;
            float *ptr = X + (n * C + c) * spatial;
            for (int i = 0; i < spatial; i++) sum += ptr[i];
            Y[n * C + c] = sum / spatial;
        }
    }
    return 0;
}
