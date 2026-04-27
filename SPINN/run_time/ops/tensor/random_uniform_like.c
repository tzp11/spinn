#include "../kernels.h"

/* RandomUniformLike: 生成与输入形状相同的均匀分布 */
int op_random_uniform_like(SpinnTensor **in, int n_in,
                           void *params, uint16_t params_size,
                           SpinnTensor **out, int n_out) {
    // 复用 RandomUniform 的逻辑
    return op_random_uniform(in, n_in, params, params_size, out, n_out);
}

/* 前向声明 */
int op_random_uniform(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
