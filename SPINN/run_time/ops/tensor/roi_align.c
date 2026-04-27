#include "../kernels.h"

/* RoiAlign: ROI对齐池化 (占位符) */
int op_roi_align(SpinnTensor **in, int n_in,
                 void *params, uint16_t params_size,
                 SpinnTensor **out, int n_out) {
    // RoiAlign 需要ROI坐标处理和双线性插值
    (void)in; (void)n_in; (void)params; (void)params_size;
    (void)out; (void)n_out;
    return 0;
}
