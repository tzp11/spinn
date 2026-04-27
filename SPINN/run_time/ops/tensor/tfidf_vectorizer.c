#include "../kernels.h"

/* TfIdfVectorizer: TF-IDF文本向量化 (占位符) */
int op_tfidf_vectorizer(SpinnTensor **in, int n_in,
                        void *params, uint16_t params_size,
                        SpinnTensor **out, int n_out) {
    // TF-IDF需要词汇表、IDF值等复杂参数
    (void)in; (void)n_in; (void)params; (void)params_size;
    (void)out; (void)n_out;
    return 0;
}
