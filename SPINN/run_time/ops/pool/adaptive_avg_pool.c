#include "../kernels.h"
#include "../../../ennf_op_params.h"
#include <math.h>

/* AdaptiveAvgPool2D: 自动适配输出尺寸的平均池化 */
int op_adaptive_avg_pool(SpinnTensor **in, int n_in,
                         void *params, uint16_t params_size,
                         SpinnTensor **out, int n_out) {
    if (n_in < 1 || n_out < 1) return -1;
    
    float *X = (float*)in[0]->data;
    float *Y = (float*)out[0]->data;
    
    int N = in[0]->dims[0];
    int C = in[0]->dims[1];
    int H_in = in[0]->dims[2];
    int W_in = in[0]->dims[3];
    
    int H_out = out[0]->dims[2];
    int W_out = out[0]->dims[3];
    
    // 如果是 1x1，走快速路径
    if (H_out == 1 && W_out == 1) {
        int spatial = H_in * W_in;
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
    
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int h_out = 0; h_out < H_out; h_out++) {
                int h_start = (int)floorf((float)(h_out * H_in) / H_out);
                int h_end   = (int)ceill((float)((h_out + 1) * H_in) / H_out);
                int h_len = h_end - h_start;
                
                for (int w_out = 0; w_out < W_out; w_out++) {
                    int w_start = (int)floorf((float)(w_out * W_in) / W_out);
                    int w_end   = (int)ceill((float)((w_out + 1) * W_in) / W_out);
                    int w_len = w_end - w_start;
                    
                    if (h_len <= 0 || w_len <= 0) {
                        Y[((n * C + c) * H_out + h_out) * W_out + w_out] = 0;
                        continue;
                    }
                    
                    float sum = 0;
                    for (int h = h_start; h < h_end; h++) {
                        for (int w = w_start; w < w_end; w++) {
                             sum += X[((n * C + c) * H_in + h) * W_in + w];
                        }
                    }
                    Y[((n * C + c) * H_out + h_out) * W_out + w_out] = sum / (h_len * w_len);
                }
            }
        }
    }
    return 0;
}
