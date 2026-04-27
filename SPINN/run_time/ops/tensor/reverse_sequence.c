#include "../kernels.h"

/* ReverseSequence: 按序列长度反转 (占位符) */
int op_reverse_sequence(SpinnTensor **in, int n_in,
                        void *params, uint16_t params_size,
                        SpinnTensor **out, int n_out) {
    // ReverseSequence 需要sequence_lens参数，较复杂
    (void)in; (void)n_in; (void)params; (void)params_size;
    (void)out; (void)n_out;
    return 0;
}
