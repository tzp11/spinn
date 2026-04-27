#include "../kernels.h"
#include "../../../ennf_op_params.h"
#include <math.h>

/* Softmax: out = exp(in) / sum(exp(in)) along axis */
int op_softmax(SpinnTensor **in, int n_in,
               void *params, uint16_t params_size,
               SpinnTensor **out, int n_out) {
    if (n_in < 1 || n_out < 1) return -1;
    
    int axis = -1;
    if (params && params_size >= sizeof(ENNF_SoftmaxParams)) {
        ENNF_SoftmaxParams *p = (ENNF_SoftmaxParams*)params;
        axis = p->axis;
    }
    
    int ndim = in[0]->ndim;
    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim) return -1;
    
    int outer = 1;
    for (int i = 0; i < axis; i++) outer *= in[0]->dims[i];
    int dim = in[0]->dims[axis];
    int inner = 1;
    for (int i = axis + 1; i < ndim; i++) inner *= in[0]->dims[i];
    
    float *x = (float*)in[0]->data;
    float *y = (float*)out[0]->data;
    
    for (int i = 0; i < outer; i++) {
        for (int k = 0; k < inner; k++) {
            int base_ptr = i * dim * inner + k;
            float max_val = x[base_ptr];
            for (int j = 1; j < dim; j++) {
                float val = x[base_ptr + j * inner];
                if (val > max_val) max_val = val;
            }
            float sum = 0;
            for (int j = 0; j < dim; j++) {
                float v = expf(x[base_ptr + j * inner] - max_val);
                y[base_ptr + j * inner] = v;
                sum += v;
            }
            if (sum == 0) sum = 1e-10f;
            float scale = 1.0f / sum;
            for (int j = 0; j < dim; j++) {
                y[base_ptr + j * inner] *= scale;
            }
        }
    }
    return 0;
}
