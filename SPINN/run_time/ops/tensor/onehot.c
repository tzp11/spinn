#include "../kernels.h"
#include "../../../ennf_op_params.h"

/* OneHot: 独热编码 */
int op_onehot(SpinnTensor **in, int n_in,
              void *params, uint16_t params_size,
              SpinnTensor **out, int n_out) {
    if (n_in < 3 || n_out < 1) return -1;
    
    // in[0]: indices, in[1]: depth (scalar), in[2]: values [off_value, on_value]
    int64_t *indices = (int64_t*)in[0]->data;
    int depth = ((int32_t*)in[1]->data)[0];
    float *values = (float*)in[2]->data;
    float *Y = (float*)out[0]->data;
    
    float off_value = values[0];
    float on_value = values[1];
    
    int axis = -1; // Default
    if (params && params_size >= sizeof(ENNF_OneHotParams)) {
        ENNF_OneHotParams *p = (ENNF_OneHotParams*)params;
        (void)p;
    }
    (void)axis;
    
    uint32_t indices_count = in[0]->elem_count;
    
    // 简化实现：假设 axis=-1
    for (uint32_t i = 0; i < indices_count; i++) {
        for (int j = 0; j < depth; j++) {
            Y[i * depth + j] = (j == indices[i]) ? on_value : off_value;
        }
    }
    return 0;
}
