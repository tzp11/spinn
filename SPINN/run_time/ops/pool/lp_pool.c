#include "../kernels.h"
#include "../../../ennf_op_params.h"
#include <math.h>

/* LpPool: Lp范数池化 */
int op_lp_pool(SpinnTensor **in, int n_in,
               void *params, uint16_t params_size,
               SpinnTensor **out, int n_out) {
    if (n_in < 1 || n_out < 1) return -1;
    
    float *X = (float*)in[0]->data;
    float *Y = (float*)out[0]->data;
    
    int p = 2; // L2 default
    int kh = 2, kw = 2;
    int stride_h = 2, stride_w = 2;
    
    if (params && params_size >= sizeof(ENNF_PoolParams)) {
        ENNF_PoolParams *pp = (ENNF_PoolParams*)params;
        kh = pp->kernel_shape[0];
        kw = pp->kernel_shape[1];
        stride_h = pp->strides[0];
        stride_w = pp->strides[1];
        p = pp->p;
    }
    
    int N = in[0]->dims[0];
    int C = in[0]->dims[1];
    int H_in = in[0]->dims[2];
    int W_in = in[0]->dims[3];
    int H_out = out[0]->dims[2];
    int W_out = out[0]->dims[3];
    
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H_out; h++) {
                for (int w = 0; w < W_out; w++) {
                    double sum = 0;
                    for (int kh_i = 0; kh_i < kh; kh_i++) {
                        for (int kw_i = 0; kw_i < kw; kw_i++) {
                            int h_idx = h * stride_h + kh_i;
                            int w_idx = w * stride_w + kw_i;
                            if (h_idx < H_in && w_idx < W_in) {
                                float val = X[((n * C + c) * H_in + h_idx) * W_in + w_idx];
                                sum += pow(fabs(val), p);
                            }
                        }
                    }
                    Y[((n * C + c) * H_out + h) * W_out + w] = pow(sum, 1.0 / p);
                }
            }
        }
    }
    return 0;
}
