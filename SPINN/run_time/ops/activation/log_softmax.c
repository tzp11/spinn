#include "../kernels.h"
#include "../../../ennf_op_params.h"
#include <math.h>

/* LogSoftmax: log(exp(x) / sum(exp(x))) = x - log(sum(exp(x))) */
int op_log_softmax(SpinnTensor **in, int n_in,
                   void *params, uint16_t params_size,
                   SpinnTensor **out, int n_out) {
    if (n_in < 1 || n_out < 1) return -1;
    
    float *X = (float*)in[0]->data;
    float *Y = (float*)out[0]->data;
    
    int axis = 1; // Default
    if (params && params_size >= sizeof(ENNF_SoftmaxParams)) {
        ENNF_SoftmaxParams *p = (ENNF_SoftmaxParams*)params;
        axis = p->axis;
    }
    
    int ndim = in[0]->ndim;
    if (axis < 0) axis += ndim;
    
    int outer = 1;
    for (int i = 0; i < axis; i++) outer *= in[0]->dims[i];
    int dim = in[0]->dims[axis];
    int inner = 1;
    for (int i = axis + 1; i < ndim; i++) inner *= in[0]->dims[i];
    
    for (int i = 0; i < outer; i++) {
        for (int k = 0; k < inner; k++) {
            // 找最大值（数值稳定性）
            float max_val = -1e30f;
            for (int j = 0; j < dim; j++) {
                float val = X[i * dim * inner + j * inner + k];
                if (val > max_val) max_val = val;
            }
            
            // 计算 sum(exp(x - max))
            float sum_exp = 0;
            for (int j = 0; j < dim; j++) {
                float val = X[i * dim * inner + j * inner + k];
                sum_exp += expf(val - max_val);
            }
            
            float log_sum = logf(sum_exp) + max_val;
            
            // log_softmax = x - log_sum
            for (int j = 0; j < dim; j++) {
                Y[i * dim * inner + j * inner + k] = 
                    X[i * dim * inner + j * inner + k] - log_sum;
            }
        }
    }
    return 0;
}
