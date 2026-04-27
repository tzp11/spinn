#include "../kernels.h"
#include "../../../ennf_op_params.h"
#include <string.h>

/* ConstantOfShape: 根据输入形状生成常量 Tensor */
int op_constant_of_shape(SpinnTensor **in, int n_in,
                         void *params, uint16_t params_size,
                         SpinnTensor **out, int n_out) {
    if (n_out < 1) return -1;
    
    float val = 0.0f;
    if (params && params_size >= sizeof(float)) {
        val = *(float*)params; // 简化的 params，假设第一个 float 是值
    }
    
    float *y = (float*)out[0]->data;
    uint32_t n = out[0]->elem_count;
    
    for (uint32_t i = 0; i < n; i++) {
        y[i] = val;
    }
    return 0;
}
