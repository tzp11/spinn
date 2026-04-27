#include "../kernels.h"
#include "../../../ennf_op_params.h"
#include <math.h>

/* LRN: Local Response Normalization */
int op_lrn(SpinnTensor **in, int n_in,
           void *params, uint16_t params_size,
           SpinnTensor **out, int n_out) {
    if (n_in < 1 || n_out < 1) return -1;
    
    float *X = (float*)in[0]->data;
    float *Y = (float*)out[0]->data;
    
    float alpha = 0.0001f;
    float beta = 0.75f;
    float bias = 1.0f;
    int size = 5;
    
    if (params && params_size >= sizeof(ENNF_LRNParams)) {
        ENNF_LRNParams *p = (ENNF_LRNParams*)params;
        alpha = p->alpha;
        beta = p->beta;
        bias = p->bias;
        size = p->size;
    }
    
    int N = in[0]->dims[0];
    int C = in[0]->dims[1];
    int H = in[0]->dims[2];
    int W = in[0]->dims[3];
    int spatial = H * W;
    
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int s = 0; s < spatial; s++) {
                // 计算局部平方和
                float sum_sq = 0;
                int c_start = (c - size / 2 < 0) ? 0 : c - size / 2;
                int c_end = (c + size / 2 >= C) ? C : c + size / 2 + 1;
                
                for (int cc = c_start; cc < c_end; cc++) {
                    float val = X[(n * C + cc) * spatial + s];
                    sum_sq += val * val;
                }
                
                float scale = powf(bias + alpha * sum_sq / size, beta);
                Y[(n * C + c) * spatial + s] = X[(n * C + c) * spatial + s] / scale;
            }
        }
    }
    return 0;
}
