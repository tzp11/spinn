#include "../kernels.h"
#include "../../../ennf_op_params.h"

/* Dropout: 推理时通常是恒等映射 */
int op_dropout(SpinnTensor **in, int n_in,
               void *params, uint16_t params_size,
               SpinnTensor **out, int n_out) {
    if (n_in < 1 || n_out < 1) return -1;
    
    // 推理模式：直接复制输入到输出
    float *X = (float*)in[0]->data;
    float *Y = (float*)out[0]->data;
    uint32_t n = in[0]->elem_count;
    
    for (uint32_t i = 0; i < n; i++) {
        Y[i] = X[i];
    }
    
    // 如果需要输出mask (通常是第二个输出)
    if (n_out >= 2 && out[1]->data) {
        // mask全1
        uint8_t *mask = (uint8_t*)out[1]->data;
        for (uint32_t i = 0; i < out[1]->elem_count; i++) {
            mask[i] = 1;
        }
    }
    
    return 0;
}
