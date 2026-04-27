#include "../kernels.h"
#include <string.h>

/* MaxUnpool: 反池化 (占位符) */
int op_max_unpool(SpinnTensor **in, int n_in,
                  void *params, uint16_t params_size,
                  SpinnTensor **out, int n_out) {
    // MaxUnpool 需要 indices 输入，较复杂
    (void)in; (void)n_in; (void)params; (void)params_size;
    
    if (n_out > 0 && out[0]->data) {
        memset(out[0]->data, 0, out[0]->size);
    }
    return 0;
}
