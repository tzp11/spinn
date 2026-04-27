#include "../kernels.h"

/* ScatterElements: 按元素scatter (简化版) */
int op_scatter_elements(SpinnTensor **in, int n_in,
                        void *params, uint16_t params_size,
                        SpinnTensor **out, int n_out) {
    // 与Scatter类似，但更通用
    return op_scatter(in, n_in, params, params_size, out, n_out);
}

int op_scatter(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
