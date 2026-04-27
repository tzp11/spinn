#include "../kernels.h"

/* Resize: 图像缩放 (简化版) */
int op_resize(SpinnTensor **in, int n_in,
              void *params, uint16_t params_size,
              SpinnTensor **out, int n_out) {
    // Resize 需要插值模式、坐标变换等，较复杂
    // 简化实现：仅nearest neighbor
    if (n_in < 1 || n_out < 1) return -1;
    
    float *X = (float*)in[0]->data;
    float *Y = (float*)out[0]->data;
    
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
                    int h_in = h * H_in / H_out;
                    int w_in = w * W_in / W_out;
                    Y[((n * C + c) * H_out + h) * W_out + w] = 
                        X[((n * C + c) * H_in + h_in) * W_in + w_in];
                }
            }
        }
    }
    return 0;
}
