#include "../kernels.h"

/* DeformableConv2D: 可变形卷积 (占位符) */
int op_deformable_conv(SpinnTensor **in, int n_in,
                       void *params, uint16_t params_size,
                       SpinnTensor **out, int n_out) {
    // 可变形卷积需要offset field，实现复杂
    (void)in; (void)n_in; (void)params; (void)params_size;
    (void)out; (void)n_out;
    return 0;
}
