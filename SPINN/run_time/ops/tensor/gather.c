#include "../kernels.h"
#include "../../../ennf_op_params.h"
int op_gather(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out) {
    if (n_in < 2 || n_out < 1) return -1;
    float *x = (float*)in[0]->data;
    int64_t *indices = (int64_t*)in[1]->data;
    float *y = (float*)out[0]->data;
    int axis = 0;
    if (params && params_size >= sizeof(ENNF_GatherParams)) axis = ((ENNF_GatherParams*)params)->axis;
    int ndim = in[0]->ndim;
    if (axis < 0) axis += ndim;
    int outer = 1; for (int i = 0; i < axis; i++) outer *= in[0]->dims[i];
    int axis_size = in[0]->dims[axis];
    int inner = 1; for (int i = axis + 1; i < ndim; i++) inner *= in[0]->dims[i];
    uint32_t num_indices = in[1]->elem_count;
    int out_idx = 0;
    for (int o = 0; o < outer; o++) {
        for (uint32_t idx = 0; idx < num_indices; idx++) {
            int64_t i = indices[idx]; if (i < 0) i += axis_size;
            int src_base = (o * axis_size + i) * inner;
            for (int k = 0; k < inner; k++) y[out_idx++] = x[src_base + k];
        }
    }
    return 0;
}
