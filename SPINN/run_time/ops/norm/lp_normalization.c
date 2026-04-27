#include "../kernels.h"
#include <math.h>

/* LpNormalization: 按轴归一化 */
int op_lp_normalization(SpinnTensor **in, int n_in,
                        void *params, uint16_t params_size,
                        SpinnTensor **out, int n_out) {
    if (n_in < 1 || n_out < 1) return -1;
    
    float *X = (float*)in[0]->data;
    float *Y = (float*)out[0]->data;
    
    int p = 2; // L2 default
    int axis = -1; // Last axis default
    int ndim = in[0]->ndim;
    
    if (axis < 0) axis += ndim;
    
    int outer = 1;
    for (int i = 0; i < axis; i++) outer *= in[0]->dims[i];
    int dim = in[0]->dims[axis];
    int inner = 1;
    for (int i = axis + 1; i < ndim; i++) inner *= in[0]->dims[i];
    
    for (int i = 0; i < outer; i++) {
        for (int k = 0; k < inner; k++) {
            // 计算 Lp norm
            double norm = 0;
            for (int j = 0; j < dim; j++) {
                float val = X[i * dim * inner + j * inner + k];
                norm += pow(fabs(val), p);
            }
            norm = pow(norm, 1.0 / p);
            
            if (norm < 1e-12) norm = 1e-12; // 避免除零
            
            // 归一化
            for (int j = 0; j < dim; j++) {
                Y[i * dim * inner + j * inner + k] = 
                    X[i * dim * inner + j * inner + k] / norm;
            }
        }
    }
    return 0;
}
