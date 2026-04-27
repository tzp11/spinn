#include "../kernels.h"

/* MaxRoiPool: ROI 最大池化 (占位符) */
int op_max_roi_pool(SpinnTensor **in, int n_in,
                    void *params, uint16_t params_size,
                    SpinnTensor **out, int n_out) {
    // MaxRoiPool 需要 RoI 坐标处理，较复杂
    (void)in; (void)n_in; (void)params; (void)params_size;
    (void)out; (void)n_out;
    return 0;
}
