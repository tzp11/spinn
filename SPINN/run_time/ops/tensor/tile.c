#include "../kernels.h"

#include <string.h>

/* Tile: 按照repeats重复张量 */
int op_tile(SpinnTensor **in, int n_in,
            void *params, uint16_t params_size,
            SpinnTensor **out, int n_out) {
    if (n_in < 2 || n_out < 1) return -1;
    
    // in[0]: input, in[1]: repeats (INT64)
    uint8_t *src = (uint8_t*)in[0]->data;
    uint8_t *dst = (uint8_t*)out[0]->data;
    // Repeats is always INT64 as per ONNX spec
    int64_t *repeats = (int64_t*)in[1]->data;
    (void)repeats;
    
    int elem_size = 4;
    if (in[0]->dtype == 7) elem_size = 8; // INT64
    else if (in[0]->dtype == 6) elem_size = 4; // INT32
    else if (in[0]->dtype == 1) elem_size = 4; // FLOAT
    else if (in[0]->dtype == 2) elem_size = 1; // UINT8
    else if (in[0]->dtype == 3) elem_size = 1; // INT8
    // Add more if needed
    
    int ndim = in[0]->ndim;
    uint32_t in_dims[8];
    uint32_t out_dims[8];
    uint32_t out_strides[8];
    uint32_t in_strides[8];
    
    for (int i = 0; i < ndim; i++) {
        in_dims[i] = in[0]->dims[i];
        out_dims[i] = out[0]->dims[i];
    }
    
    // Calculate strides
    out_strides[ndim - 1] = 1;
    in_strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        out_strides[i] = out_strides[i + 1] * out_dims[i + 1];
        in_strides[i]  = in_strides[i + 1] * in_dims[i + 1];
    }
    
    uint32_t n = out[0]->elem_count;
    
    for (uint32_t i = 0; i < n; i++) {
        uint32_t tmp = i;
        uint32_t in_offset = 0;
        
        for (int d = 0; d < ndim; d++) {
            uint32_t coord = tmp / out_strides[d];
            tmp %= out_strides[d];
            
            // Map to input coordinate: coord % in_dim
            uint32_t in_coord = coord % in_dims[d];
            in_offset += in_coord * in_strides[d];
        }
        
        // Copy element
        memcpy(dst + i * elem_size, src + in_offset * elem_size, elem_size);
    }
    
    return 0;
}
