#include "../kernels.h"
#include "../../../ennf_op_params.h"
#include <string.h>
int op_concat(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out) {
    if (n_in < 1 || n_out < 1) return -1;
    int axis = 0;
    if (params && params_size >= sizeof(ENNF_ConcatParams)) axis = ((ENNF_ConcatParams*)params)->axis;
    int ndim = in[0]->ndim;
    if (axis < 0) axis += ndim;
    float *y = (float*)out[0]->data;
    int outer = 1; for (int i = 0; i < axis; i++) outer *= in[0]->dims[i];
    int inner = 1; for (int i = axis + 1; i < ndim; i++) inner *= in[0]->dims[i];
    int out_offset = 0;
    for (int o = 0; o < outer; o++) {
        for (int t = 0; t < n_in; t++) {
            float *x = (float*)in[t]->data;
            int axis_size = in[t]->dims[axis];
            int copy_size = axis_size * inner;
            memcpy(y + out_offset, x + o * axis_size * inner, copy_size * sizeof(float));
            out_offset += copy_size;
        }
    }
    return 0;
}
