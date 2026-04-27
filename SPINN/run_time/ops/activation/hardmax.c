#include "../kernels.h"
#include <string.h>

/* Hardmax: 将最大元素设为1，其余为0 */
int op_hardmax(SpinnTensor **in, int n_in,
               void *params, uint16_t params_size,
               SpinnTensor **out, int n_out) {
    if (n_in < 1 || n_out < 1) return -1;
    
    float *X = (float*)in[0]->data;
    float *Y = (float*)out[0]->data;
    
    int axis = 1; // Default
    int ndim = in[0]->ndim;
    
    // 参数解析省略，假设 axis=1
    
    int outer = 1;
    for (int i = 0; i < axis; i++) outer *= in[0]->dims[i];
    int dim = in[0]->dims[axis];
    int inner = 1;
    for (int i = axis + 1; i < ndim; i++) inner *= in[0]->dims[i];
    
    for (int i = 0; i < outer; i++) {
        for (int k = 0; k < inner; k++) {
            // 找到最大值的索引
            float max_val = -1e30f;
            int max_idx = 0;
            
            for (int j = 0; j < dim; j++) {
                float val = X[i * dim * inner + j * inner + k];
                if (val > max_val) {
                    max_val = val;
                    max_idx = j;
                }
            }
            
            // 设置输出
            for (int j = 0; j < dim; j++) {
                Y[i * dim * inner + j * inner + k] = (j == max_idx) ? 1.0f : 0.0f;
            }
        }
    }
    return 0;
}
