#include "../kernels.h"
#include "../../../ennf_op_params.h"
#include <string.h>

/* ReduceMean */
int op_reduce_mean(SpinnTensor **in, int n_in,
                   void *params, uint16_t params_size,
                   SpinnTensor **out, int n_out) {
    if (n_in < 1 || n_out < 1) return -1;
    
    int axis = -1;
    int merge_axes = 0;
    
    if (params && params_size >= sizeof(ENNF_ReduceParams)) {
        ENNF_ReduceParams *p = (ENNF_ReduceParams*)params;
        if (p->num_axes > 0) {
            axis = p->axes[0];
            if (axis < 0) axis += in[0]->ndim;
            if (p->num_axes > 1 && axis + p->num_axes == in[0]->ndim) merge_axes = 1;
        }
    }
    
    float *x = (float*)in[0]->data;
    float *y = (float*)out[0]->data;
    int ndim = in[0]->ndim;
    
    if (axis == -1 && (!params || ((ENNF_ReduceParams*)params)->num_axes == 0)) {
        if (out[0]->elem_count > 1 && in[0]->ndim == out[0]->ndim + 1) {
            int inferred_axis = -1;
            int out_d = 0;
            for (int d = 0; d < in[0]->ndim; d++) {
                if (out_d < out[0]->ndim && in[0]->dims[d] == out[0]->dims[out_d]) out_d++;
                else { if (inferred_axis == -1) inferred_axis = d; else { inferred_axis = -2; break; } }
            }
            if (inferred_axis >= 0) axis = inferred_axis;
        } else if (in[0]->ndim == out[0]->ndim) {
            int first_diff = -1, diff_cnt = 0;
            for(int d=0; d<in[0]->ndim; d++) {
                if(in[0]->dims[d] != out[0]->dims[d]) {
                    if(out[0]->dims[d] == 1) { if(first_diff == -1) first_diff = d; diff_cnt++; }
                    else { first_diff = -2; break; }
                } else if (first_diff != -1) { first_diff = -2; break; }
            }
            if (first_diff >= 0 && first_diff + diff_cnt == in[0]->ndim) { axis = first_diff; merge_axes = 1; }
        }
    }

    if (axis == -1) {
        uint32_t n = in[0]->elem_count;
        float sum = 0;
        for (uint32_t i = 0; i < n; i++) sum += x[i];
        if (out[0]->elem_count == 1) y[0] = sum / n;
        else for (uint32_t i = 0; i < out[0]->elem_count; i++) y[i] = sum / n;
        return 0;
    }
    
    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim) return -1;
    
    int outer = 1;
    for (int i = 0; i < axis; i++) outer *= in[0]->dims[i];
    int dim = in[0]->dims[axis];
    int inner = 1;
    if (merge_axes) { for (int i = axis + 1; i < ndim; i++) dim *= in[0]->dims[i]; inner = 1; }
    else { for (int i = axis + 1; i < ndim; i++) inner *= in[0]->dims[i]; }
    
    memset(y, 0, out[0]->size);
    for (int i = 0; i < outer; i++) {
        for (int k = 0; k < inner; k++) {
            float sum = 0;
            for (int j = 0; j < dim; j++) sum += x[i * dim * inner + j * inner + k];
            y[i * inner + k] = sum / dim;
        }
    }
    return 0;
}
