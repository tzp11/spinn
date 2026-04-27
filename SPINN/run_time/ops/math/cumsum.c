#include "../kernels.h"
#include "../../../ennf_op_params.h"

/* CumSum: 沿轴累积求和 */
int op_cumsum(SpinnTensor **in, int n_in,
              void *params, uint16_t params_size,
              SpinnTensor **out, int n_out) {
    if (n_in < 2 || n_out < 1) return -1;
    
    // in[0]: data, in[1]: axis (scalar)
    float *x = (float*)in[0]->data;
    float *y = (float*)out[0]->data;
    
    int axis = 0;
    if (in[1]->data) {
        if (in[1]->dtype == ENNF_TYPE_FLOAT32) axis = (int)((float*)in[1]->data)[0];
        else axis = ((int32_t*)in[1]->data)[0];
    }
    
    int ndim = in[0]->ndim;
    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim) return -1;
    
    int outer = 1;
    for (int i = 0; i < axis; i++) outer *= in[0]->dims[i];
    int dim = in[0]->dims[axis];
    int inner = 1;
    for (int i = axis + 1; i < ndim; i++) inner *= in[0]->dims[i];
    
    for (int i = 0; i < outer; i++) {
        for (int k = 0; k < inner; k++) {
            float sum = 0;
            for (int j = 0; j < dim; j++) {
                float v = x[i * dim * inner + j * inner + k];
                sum += v;
                y[i * dim * inner + j * inner + k] = sum;
            }
        }
    }
    return 0;
}
