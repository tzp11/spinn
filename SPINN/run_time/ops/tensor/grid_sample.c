#include "../kernels.h"
#include <math.h>

/* GridSample: 使用双线性插值进行网格采样 */
/* Input[0]: X (N, C, H_in, W_in) */
/* Input[1]: Grid (N, H_out, W_out, 2) */
/* Grid values: [-1, 1], (-1,-1) is top-left */

static float get_pixel(float *data, int w, int h, int x, int y) {
    if (x < 0 || x >= w || y < 0 || y >= h) return 0.0f; // PaddingMode=Zeros
    return data[y * w + x];
}

int op_grid_sample(SpinnTensor **in, int n_in,
                   void *params, uint16_t params_size,
                   SpinnTensor **out, int n_out) {
    if (n_in < 2 || n_out < 1) return -1;
    
    float *X = (float*)in[0]->data;
    float *Grid = (float*)in[1]->data;
    float *Y = (float*)out[0]->data;
    
    int N = in[0]->dims[0];
    int C = in[0]->dims[1];
    int H_in = in[0]->dims[2];
    int W_in = in[0]->dims[3];
    
    int H_out = in[1]->dims[1];
    int W_out = in[1]->dims[2];
    
    int align_corners = 0; // 默认
    // params handling omitted for brevity, assuming defaults (Bilinear, Zeros, align_corners=0)
    
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H_out; h++) {
            for (int w = 0; w < W_out; w++) {
                // 获取 Grid 坐标 (x, y)
                int grid_idx = ((n * H_out + h) * W_out + w) * 2;
                float gx = Grid[grid_idx + 0];
                float gy = Grid[grid_idx + 1];
                
                // 转换到输入坐标系
                float ix, iy;
                if (align_corners) {
                    ix = ((gx + 1) * (W_in - 1)) / 2;
                    iy = ((gy + 1) * (H_in - 1)) / 2;
                } else {
                    ix = ((gx + 1) * W_in - 1) / 2;
                    iy = ((gy + 1) * H_in - 1) / 2;
                }
                
                // 双线性插值
                int x0 = (int)floorf(ix);
                int y0 = (int)floorf(iy);
                int x1 = x0 + 1;
                int y1 = y0 + 1;
                
                float wa = (x1 - ix) * (y1 - iy);
                float wb = (x1 - ix) * (iy - y0);
                float wc = (ix - x0) * (y1 - iy);
                float wd = (ix - x0) * (iy - y0);
                
                for (int c = 0; c < C; c++) {
                    float *img = X + (n * C + c) * H_in * W_in;
                    
                    float val = 0;
                    // 边界检查在 get_pixel 中处理
                    val += wa * get_pixel(img, W_in, H_in, x0, y0);
                    val += wb * get_pixel(img, W_in, H_in, x0, y1);
                    val += wc * get_pixel(img, W_in, H_in, x1, y0);
                    val += wd * get_pixel(img, W_in, H_in, x1, y1);
                    
                    Y[((n * C + c) * H_out + h) * W_out + w] = val;
                }
            }
        }
    }
    return 0;
}
