#include "../kernels.h"
#include "../../../ennf_op_params.h"

/* SpaceToDepth: 将空间维度重排到深度维度 */
int op_space_to_depth(SpinnTensor **in, int n_in,
                      void *params, uint16_t params_size,
                      SpinnTensor **out, int n_out) {
    if (n_in < 1 || n_out < 1) return -1;
    
    float *X = (float*)in[0]->data;
    float *Y = (float*)out[0]->data;
    
    int blocksize = 2;
    if (params && params_size >= sizeof(ENNF_SpaceParams)) {
        ENNF_SpaceParams *p = (ENNF_SpaceParams*)params;
        blocksize = p->blocksize;
    }
    
    int N = in[0]->dims[0];
    int C = in[0]->dims[1];
    int H = in[0]->dims[2];
    int W = in[0]->dims[3];
    
    int C_out = C * blocksize * blocksize;
    int H_out = H / blocksize;
    int W_out = W / blocksize;
    
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H_out; h++) {
                for (int w = 0; w < W_out; w++) {
                    for (int bh = 0; bh < blocksize; bh++) {
                        for (int bw = 0; bw < blocksize; bw++) {
                            int h_in = h * blocksize + bh;
                            int w_in = w * blocksize + bw;
                            int c_out = c * blocksize * blocksize + bh * blocksize + bw;
                            
                            Y[((n * C_out + c_out) * H_out + h) * W_out + w] = 
                                X[((n * C + c) * H + h_in) * W + w_in];
                        }
                    }
                }
            }
        }
    }
    return 0;
}
