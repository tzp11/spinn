#include "../kernels.h"
#include "../../../ennf_op_params.h"

/* MaxPool2D */
int op_maxpool(SpinnTensor **in, int n_in,
               void *params, uint16_t params_size,
               SpinnTensor **out, int n_out) {
    if (n_in < 1 || n_out < 1) return -1;
    
    float *X = (float*)in[0]->data;
    float *Y = (float*)out[0]->data;
    
    int N = in[0]->dims[0];
    int C = in[0]->dims[1];
    int H = in[0]->dims[2];
    int W_dim = in[0]->dims[3];
    int OH = out[0]->dims[2];
    int OW = out[0]->dims[3];
    
    int KH = 1, KW = 1, SH = 1, SW = 1, PH = 0, PW = 0;
    if (params && params_size >= sizeof(ENNF_PoolParams)) {
        ENNF_PoolParams *p = (ENNF_PoolParams*)params;
        KH = p->kernel_shape[0]; KW = p->kernel_shape[1];
        SH = p->strides[0]; SW = p->strides[1];
        PH = p->pads[0]; PW = p->pads[1];
    }
    
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    float max_val = -1e30f;
                    for (int kh = 0; kh < KH; kh++) {
                        for (int kw = 0; kw < KW; kw++) {
                            int ih = oh * SH - PH + kh;
                            int iw = ow * SW - PW + kw;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W_dim) {
                                float val = X[((n * C + c) * H + ih) * W_dim + iw];
                                if (val > max_val) max_val = val;
                            }
                        }
                    }
                    Y[((n * C + c) * OH + oh) * OW + ow] = max_val;
                }
            }
        }
    }
    return 0;
}
