#include "../kernels.h"

/* RandomNormalLike: 生成与输入形状相同的正态分布 */
int op_random_normal_like(SpinnTensor **in, int n_in,
                          void *params, uint16_t params_size,
                          SpinnTensor **out, int n_out) {
    // 复用 RandomNormal 的逻辑
    return op_random_normal(in, n_in, params, params_size, out, n_out);
}

/* 前向声明 */
int op_random_normal(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
