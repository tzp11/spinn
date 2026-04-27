#include "../kernels.h"
#include <string.h>

/* EyeLike: 创建与输入形状相同的单位矩阵 */
int op_eyelike(SpinnTensor **in, int n_in,
               void *params, uint16_t params_size,
               SpinnTensor **out, int n_out) {
    if (n_in < 1 || n_out < 1) return -1;
    
    float *Y = (float*)out[0]->data;
    
    // 初始化为0
    memset(Y, 0, out[0]->size);
    
    // 假设是2D矩阵
    int rows = out[0]->dims[0];
    int cols = (out[0]->ndim >= 2) ? out[0]->dims[1] : 1;
    
    int k = 0; // 对角线偏移
    // k可以从params读取，这里简化为0
    
    for (int i = 0; i < rows; i++) {
        int j = i + k;
        if (j >= 0 && j < cols) {
            Y[i * cols + j] = 1.0f;
        }
    }
    
    return 0;
}
