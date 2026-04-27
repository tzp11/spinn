#include "../kernels.h"
#include "../../../ennf_op_params.h"

/* GatherElements: 按元素索引聚集 */
int op_gather_elements(SpinnTensor **in, int n_in,
                       void *params, uint16_t params_size,
                       SpinnTensor **out, int n_out) {
    if (n_in < 2 || n_out < 1) return -1;
    
    float *data = (float*)in[0]->data;
    float *Y = (float*)out[0]->data;
    
    int axis = 0;
    if (params && params_size >= sizeof(ENNF_GatherParams)) {
        ENNF_GatherParams *p = (ENNF_GatherParams*)params;
        axis = p->axis;
    }
    
    int ndim = in[0]->ndim;
    if (axis < 0) axis += ndim;
    
    // stride calculation based on OUTPUT dims (because we iterate over output)
    uint32_t out_strides[8];
    uint32_t in_strides[8];
    out_strides[ndim - 1] = 1;
    in_strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        out_strides[i] = out_strides[i + 1] * out[0]->dims[i + 1];
        in_strides[i]  = in_strides[i + 1] * in[0]->dims[i + 1];
    }
    
    uint32_t n = out[0]->elem_count;
    void *indices_raw = in[1]->data;
    uint8_t indices_type = in[1]->dtype; // 6=INT32, 7=INT64

    for (uint32_t idx = 0; idx < n; idx++) {
        // Calculate output coordinates
        uint32_t coords[8];
        uint32_t tmp = idx;
        for (int d = 0; d < ndim; d++) {
            coords[d] = tmp / out_strides[d];
            tmp = tmp % out_strides[d];
        }
        
        // Get gather index from indices tensor (handle dtype)
        int64_t gather_idx = 0;
        if (indices_type == 7) { // INT64
            gather_idx = ((int64_t*)indices_raw)[idx];
        } else { // Assume INT32 (most common for indices besides INT64)
            gather_idx = ((int32_t*)indices_raw)[idx];
        }

        // Handle negative indices
        if (gather_idx < 0) gather_idx += in[0]->dims[axis];
        
        // Safety check for index out of bounds
        if (gather_idx < 0 || gather_idx >= in[0]->dims[axis]) {
             // In a real runtime we might want to error out or clamp
             // fprintf(stderr, "GatherElements: Index %ld out of bounds for axis %d dim %d\n", gather_idx, axis, in[0]->dims[axis]);
             gather_idx = 0; // Fallback to avoid crash
        }

        // Replace coordinate at 'axis' with gather_idx
        coords[axis] = gather_idx;
        
        // Calculate input linearized index
        uint32_t data_idx = 0;
        for (int d = 0; d < ndim; d++) {
            data_idx += coords[d] * in_strides[d];
        }
        
        Y[idx] = data[data_idx];
    }
    
    return 0;
}
