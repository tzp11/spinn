#include "../kernels.h"
#include <string.h>
int op_flatten(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out) {
    (void)params; (void)params_size;
    if (n_in < 1 || n_out < 1) return -1;
    if (in[0]->data != out[0]->data) memcpy(out[0]->data, in[0]->data, in[0]->size);
    return 0;
}
