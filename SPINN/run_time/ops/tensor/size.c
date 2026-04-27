#include "../kernels.h"

/* Size: 返回张量的元素总数 */
int op_size(SpinnTensor **in, int n_in,
            void *params, uint16_t params_size,
            SpinnTensor **out, int n_out) {
    (void)params; (void)params_size;
    if (n_in < 1 || n_out < 1) return -1;
    
    int64_t *Y = (int64_t*)out[0]->data;
    Y[0] = in[0]->elem_count;
    return 0;
}
