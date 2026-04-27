#include "../kernels.h"
#include "../../../ennf_op_params.h"

/* AveragePool2D */
int op_avgpool(SpinnTensor **in, int n_in,
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
    
    int KH = 1, KW = 1, SH = 1, SW = 1, PH = 0, PW = 0, count_include_pad = 0;
    if (params && params_size >= sizeof(ENNF_PoolParams)) {
        ENNF_PoolParams *p = (ENNF_PoolParams*)params;
        KH = p->kernel_shape[0]; KW = p->kernel_shape[1];
        SH = p->strides[0]; SW = p->strides[1];
        PH = p->pads[0]; PW = p->pads[1];
        count_include_pad = p->count_include_pad;
    }
    
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < OH; oh++) {
                for (int ow = 0; ow < OW; ow++) {
                    float sum = 0; int count = 0;
                    for (int kh = 0; kh < KH; kh++) {
                        for (int kw = 0; kw < KW; kw++) {
                            int ih = oh * SH - PH + kh;
                            int iw = ow * SW - PW + kw;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W_dim) {
                                sum += X[((n * C + c) * H + ih) * W_dim + iw];
                                count++;
                            } else if (count_include_pad) { count++; }
                        }
                    }
                    Y[((n * C + c) * OH + oh) * OW + ow] = (count > 0) ? sum / count : 0;
                }
            }
        }
    }
    return 0;
}
