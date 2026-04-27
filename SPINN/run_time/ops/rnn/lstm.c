#include "../kernels.h"

/* LSTM: Long Short-Term Memory (占位符) */
int op_lstm(SpinnTensor **in, int n_in,
            void *params, uint16_t params_size,
            SpinnTensor **out, int n_out) {
    // LSTM 实现极其复杂，涉及时间步循环和多个门控单元
    (void)in; (void)n_in; (void)params; (void)params_size;
    (void)out; (void)n_out;
    return 0;
}
