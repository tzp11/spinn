#include "../kernels.h"
int op_argmin(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out) {
    if (n_in < 1 || n_out < 1) return -1;
    float *x = (float*)in[0]->data;
    int64_t *y = (int64_t*)out[0]->data;
    int axis = 0;
    if (params && params_size >= 4) axis = *(int32_t*)params;
    int ndim = in[0]->ndim;
    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim) return -1;
    int outer = 1; for (int i = 0; i < axis; i++) outer *= in[0]->dims[i];
    int dim = in[0]->dims[axis];
    int inner = 1; for (int i = axis + 1; i < ndim; i++) inner *= in[0]->dims[i];
    for (int i = 0; i < outer; i++) {
        for (int k = 0; k < inner; k++) {
            int base = i * dim * inner + k;
            float min_val = x[base]; int min_idx = 0;
            for (int j = 1; j < dim; j++) { float v = x[base + j * inner]; if (v < min_val) { min_val = v; min_idx = j; } }
            y[i * inner + k] = min_idx;
        }
    }
    return 0;
}
