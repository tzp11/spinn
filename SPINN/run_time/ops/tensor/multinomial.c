#include "../kernels.h"

/* Multinomial: 多项式采样 (占位符) */
int op_multinomial(SpinnTensor **in, int n_in,
                   void *params, uint16_t params_size,
                   SpinnTensor **out, int n_out) {
    // 需要随机数生成器
    (void)in; (void)n_in; (void)params; (void)params_size;
    (void)out; (void)n_out;
    return 0;
}
