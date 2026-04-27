#include "../kernels.h"
#include "../../../ennf_op_params.h"
#include <string.h>

/* Pad: 张量填充 */
int op_pad(SpinnTensor **in, int n_in,
           void *params, uint16_t params_size,
           SpinnTensor **out, int n_out) {
    if (n_in < 1 || n_out < 1) return -1;
    
    float *X = (float*)in[0]->data;
    float *Y = (float*)out[0]->data;
    
    // 默认 constant mode, value=0
    float constant_value = 0.0f;
    int mode = 0; // 0=constant
    
    if (params && params_size >= sizeof(ENNF_PadParams)) {
        ENNF_PadParams *p = (ENNF_PadParams*)params;
        constant_value = p->constant_value;
    }
    (void)mode;
    
    // 简化实现：仅支持2D/4D的constant padding
    // 完整实现需要从inputs[1]读取pads
    
    // 初始化输出为constant_value
    uint32_t out_size = out[0]->elem_count;
    for (uint32_t i = 0; i < out_size; i++) {
        Y[i] = constant_value;
    }
    
    // 复制输入到中心区域 (简化版本，假设输入/输出维度已知)
    // 这里仅作占位符，完整实现需要解析pads参数
    memcpy(Y, X, in[0]->elem_count * sizeof(float));
    
    return 0;
}
