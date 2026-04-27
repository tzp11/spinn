#include "../kernels.h"

/* EmbeddingBag: 嵌入包聚合 (占位符) */
int op_embedding_bag(SpinnTensor **in, int n_in,
                     void *params, uint16_t params_size,
                     SpinnTensor **out, int n_out) {
    // EmbeddingBag需要处理indices和offsets，较复杂
    (void)in; (void)n_in; (void)params; (void)params_size;
    (void)out; (void)n_out;
    return 0;
}
