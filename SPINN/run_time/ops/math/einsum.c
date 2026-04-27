#include "../kernels.h"

/* Einsum: 爱因斯坦求和约定 (占位符) */
int op_einsum(SpinnTensor **in, int n_in,
              void *params, uint16_t params_size,
              SpinnTensor **out, int n_out) {
    // Einsum需要解析equation字符串，非常复杂
    // 这里仅返回0作为占位符
    (void)in; (void)n_in; (void)params; (void)params_size;
    (void)out; (void)n_out;
    return 0;
}
