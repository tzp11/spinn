#include "../kernels.h"
#include "../../../ennf_op_params.h"
#include <string.h>
/* Cast: 数据类型转换 */
int op_cast(SpinnTensor **in, int n_in, 
            void *params, uint16_t params_size, 
            SpinnTensor **out, int n_out) {
    if (n_in < 1 || n_out < 1) return -1;
    
    SpinnTensor *t_in = in[0];
    SpinnTensor *t_out = out[0];
    
    int src_type = t_in->dtype;
    int dst_type = t_out->dtype;
    
    uint32_t n = t_out->elem_count;
    
    if (src_type == dst_type) {
        memcpy(t_out->data, t_in->data, t_in->size);
        return 0;
    }
    
    // INT64 -> FLOAT
    if (src_type == 7 && dst_type == 1) {
        int64_t *src = (int64_t*)t_in->data;
        float *dst = (float*)t_out->data;
        for (uint32_t i = 0; i < n; i++) dst[i] = (float)src[i];
    }
    // INT32 -> FLOAT
    else if (src_type == 6 && dst_type == 1) {
        int32_t *src = (int32_t*)t_in->data;
        float *dst = (float*)t_out->data;
        for (uint32_t i = 0; i < n; i++) dst[i] = (float)src[i];
    }
    // FLOAT -> INT64
    else if (src_type == 1 && dst_type == 7) {
        float *src = (float*)t_in->data;
        int64_t *dst = (int64_t*)t_out->data;
        for (uint32_t i = 0; i < n; i++) dst[i] = (int64_t)src[i];
    }
    // FLOAT -> INT32
    else if (src_type == 1 && dst_type == 6) {
        float *src = (float*)t_in->data;
        int32_t *dst = (int32_t*)t_out->data;
        for (uint32_t i = 0; i < n; i++) dst[i] = (int32_t)src[i];
    }
    // INT64 -> INT32
    else if (src_type == 7 && dst_type == 6) {
        int64_t *src = (int64_t*)t_in->data;
        int32_t *dst = (int32_t*)t_out->data;
        for (uint32_t i = 0; i < n; i++) dst[i] = (int32_t)src[i];
    }
    // INT32 -> INT64
    else if (src_type == 6 && dst_type == 7) {
        int32_t *src = (int32_t*)t_in->data;
        int64_t *dst = (int64_t*)t_out->data;
        for (uint32_t i = 0; i < n; i++) dst[i] = (int64_t)src[i];
    }
    else {
        // Fallback for types we missed (e.g. BOOL, DOUBLE)
        return -2;
    }
    
    return 0;
}
