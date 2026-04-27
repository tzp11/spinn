#include "../kernels.h"

/* If: 控制流 (条件执行子图) */
int op_if(SpinnTensor **in, int n_in,
          void *params, uint16_t params_size,
          SpinnTensor **out, int n_out) {
    // If 需要递归调用图执行器，SpinnRuntime 目前是扁平化的。
    // 这需要架构变更来支持 subgraphs。
    (void)in; (void)n_in; (void)params; (void)params_size;
    (void)out; (void)n_out;
    return 0;
}
