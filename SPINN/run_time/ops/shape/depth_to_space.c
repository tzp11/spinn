#include "../kernels.h"
#include "../../../ennf_op_params.h"

/* DepthToSpace: 将深度维度重排到空间维度 */
int op_depth_to_space(SpinnTensor **in, int n_in,
                      void *params, uint16_t params_size,
                      SpinnTensor **out, int n_out) {
    if (n_in < 1 || n_out < 1) return -1;
    
    float *X = (float*)in[0]->data;
    float *Y = (float*)out[0]->data;
    
    int blocksize = 2;
    int mode = 0; // 0=DCR, 1=CRD
    
    if (params && params_size >= sizeof(ENNF_SpaceParams)) {
        ENNF_SpaceParams *p = (ENNF_SpaceParams*)params;
        blocksize = p->blocksize;
        mode = p->mode;
    }
    
    int N = in[0]->dims[0];
    int C = in[0]->dims[1];
    int H = in[0]->dims[2];
    int W = in[0]->dims[3];
    
    int C_out = C / (blocksize * blocksize);
    int H_out = H * blocksize;
    int W_out = W * blocksize;
    
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C_out; c++) {
            for (int h = 0; h < H_out; h++) {
                for (int w = 0; w < W_out; w++) {
                    int h_in = h / blocksize;
                    int w_in = w / blocksize;
                    int h_offset = h % blocksize;
                    int w_offset = w % blocksize;
                    
                    int c_in;
                    if (mode == 0) { // DCR
                        c_in = c * blocksize * blocksize + h_offset * blocksize + w_offset;
                    } else { // CRD
                        c_in = (h_offset * blocksize + w_offset) * C_out + c;
                    }
                    
                    Y[((n * C_out + c) * H_out + h) * W_out + w] = 
                        X[((n * C + c_in) * H + h_in) * W + w_in];
                }
            }
        }
    }
    return 0;
}
