#include "../kernels.h"

/* NonMaxSuppression: NMS后处理 (占位符) */
int op_non_max_suppression(SpinnTensor **in, int n_in,
                           void *params, uint16_t params_size,
                           SpinnTensor **out, int n_out) {
    // NMS 需要复杂的框排序和IoU计算
    (void)in; (void)n_in; (void)params; (void)params_size;
    (void)out; (void)n_out;
    return 0;
}
