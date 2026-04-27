#include "../kernels.h"
#include <math.h>

/* SoftmaxCrossEntropyLoss: 计算softmax + 交叉熵损失 */
int op_softmax_cross_entropy_loss(SpinnTensor **in, int n_in,
                                   void *params, uint16_t params_size,
                                   SpinnTensor **out, int n_out) {
    if (n_in < 2 || n_out < 1) return -1;
    
    // in[0]: scores (N, C), in[1]: labels (N) or (N, C)
    float *scores = (float*)in[0]->data;
    int64_t *labels = (int64_t*)in[1]->data;
    float *Y = (float*)out[0]->data;
    
    int N = in[0]->dims[0];
    int C = in[0]->dims[1];
    
    double total_loss = 0.0;
    
    for (int n = 0; n < N; n++) {
        float *score = scores + n * C;
        
        // 计算softmax (数值稳定版)
        float max_score = score[0];
        for (int c = 1; c < C; c++) {
            if (score[c] > max_score) max_score = score[c];
        }
        
        double sum_exp = 0.0;
        for (int c = 0; c < C; c++) {
            sum_exp += expf(score[c] - max_score);
        }
        
        // 计算交叉熵
        int label = labels[n];
        float log_prob = (score[label] - max_score) - logf(sum_exp);
        total_loss -= log_prob;
    }
    
    Y[0] = total_loss / N; // 平均损失
    return 0;
}
