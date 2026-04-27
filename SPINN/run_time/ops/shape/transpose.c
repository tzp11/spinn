#include "../kernels.h"
#include "../../../ennf_op_params.h"

/* Transpose */
int op_transpose(SpinnTensor **in, int n_in,
                 void *params, uint16_t params_size,
                 SpinnTensor **out, int n_out) {
    if (n_in < 1 || n_out < 1) return -1;
    
    float *src = (float*)in[0]->data;
    float *dst = (float*)out[0]->data;
    int ndim = in[0]->ndim;
    uint32_t total = in[0]->elem_count;
    
    uint8_t perm[ENNF_MAX_DIMS] = {0};
    if (params && params_size >= sizeof(ENNF_TransposeParams)) {
        ENNF_TransposeParams *p = (ENNF_TransposeParams*)params;
        for (int i = 0; i < ndim; i++) perm[i] = p->perm[i];
    } else {
        for (int i = 0; i < ndim; i++) perm[i] = ndim - 1 - i;
    }
    
    uint32_t in_strides[ENNF_MAX_DIMS];
    in_strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) in_strides[i] = in_strides[i + 1] * in[0]->dims[i + 1];
    
    uint32_t out_strides[ENNF_MAX_DIMS];
    out_strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) out_strides[i] = out_strides[i + 1] * out[0]->dims[i + 1];
    
    for (uint32_t idx = 0; idx < total; idx++) {
        uint32_t coords[ENNF_MAX_DIMS];
        uint32_t tmp = idx;
        for (int d = 0; d < ndim; d++) { coords[d] = tmp / in_strides[d]; tmp = tmp % in_strides[d]; }
        uint32_t out_idx = 0;
        for (int d = 0; d < ndim; d++) out_idx += coords[perm[d]] * out_strides[d];
        dst[out_idx] = src[idx];
    }
    return 0;
}
