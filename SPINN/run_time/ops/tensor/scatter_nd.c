#include "../kernels.h"

/* ScatterND: N维scatter (占位符) */
int op_scatter_nd(SpinnTensor **in, int n_in,
                  void *params, uint16_t params_size,
                  SpinnTensor **out, int n_out) {
    // ScatterND 需要处理多维索引，较复杂
    (void)in; (void)n_in; (void)params; (void)params_size;
    (void)out; (void)n_out;
    return 0;
}
