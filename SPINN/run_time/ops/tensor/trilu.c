#include "../kernels.h"
#include <string.h>

/* Trilu: 提取三角矩阵 */
int op_trilu(SpinnTensor **in, int n_in,
             void *params, uint16_t params_size,
             SpinnTensor **out, int n_out) {
    if (n_in < 1 || n_out < 1) return -1;
    
    float *X = (float*)in[0]->data;
    float *Y = (float*)out[0]->data;
    
    int upper = 1; // Default: upper triangle
    int k = 0;     // Default: main diagonal
    
    // params parsing omitted
    
    int ndim = in[0]->ndim;
    int rows = in[0]->dims[ndim - 2];
    int cols = in[0]->dims[ndim - 1];
    int num_matrices = in[0]->elem_count / (rows * cols);
    
    for (int m = 0; m < num_matrices; m++) {
        float *mat_in = X + m * rows * cols;
        float *mat_out = Y + m * rows * cols;
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (upper) {
                    // Upper triangle: keep if j >= i + k
                    mat_out[i * cols + j] = (j >= i + k) ? mat_in[i * cols + j] : 0.0f;
                } else {
                    // Lower triangle: keep if j <= i + k
                    mat_out[i * cols + j] = (j <= i + k) ? mat_in[i * cols + j] : 0.0f;
                }
            }
        }
    }
    return 0;
}
