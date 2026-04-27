#include "../kernels.h"
#include "../../../ennf_op_params.h"
#include <string.h>

/* Reshape */
int op_reshape(SpinnTensor **in, int n_in,
               void *params, uint16_t params_size,
               SpinnTensor **out, int n_out) {
    (void)params; (void)params_size;
    if (n_in < 1 || n_out < 1) return -1;
    if (in[0]->data == out[0]->data) return 0;
    memcpy(out[0]->data, in[0]->data, in[0]->size);
    return 0;
}
