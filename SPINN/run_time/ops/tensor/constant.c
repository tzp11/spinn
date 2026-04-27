#include "../kernels.h"
#include <string.h>

/* Constant: 输出存储在参数中的值 */
/* 直接将参数作为 Tensor 数据输出 (如果预分配了) 或者复制 */
int op_constant(SpinnTensor **in, int n_in,
                void *params, uint16_t params_size,
                SpinnTensor **out, int n_out) {
    if (n_out < 1) return -1;
    
    // 如果 params 存在，且输出有空间，复制数据
    if (params && params_size > 0 && out[0]->data) {
        size_t copy_size = params_size;
        if (copy_size > out[0]->size) copy_size = out[0]->size;
        memcpy(out[0]->data, params, copy_size);
    }
    return 0;
}
