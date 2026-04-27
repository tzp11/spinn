#include "../kernels.h"
#include "../../../ennf_op_params.h"
#include <math.h>

/* GlobalLpPool: 空间维度的 Lp 范数 */
int op_global_lp_pool(SpinnTensor **in, int n_in,
                      void *params, uint16_t params_size,
                      SpinnTensor **out, int n_out) {
    if (n_in < 1 || n_out < 1) return -1;
    
    float *X = (float*)in[0]->data;
    float *Y = (float*)out[0]->data;
    
    int p = 2; // 默认 L2
    if (params && params_size >= sizeof(ENNF_PoolParams)) {
        ENNF_PoolParams *pp = (ENNF_PoolParams*)params;
        p = pp->p;
    }
    
    int N = in[0]->dims[0];
    int C = in[0]->dims[1];
    
    // 计算空间大小 (H*W*...)
    int spatial = 1;
    for (int i = 2; i < in[0]->ndim; i++) spatial *= in[0]->dims[i];
    
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            double sum = 0;
            float *ptr = X + (n * C + c) * spatial;
            for (int i = 0; i < spatial; i++) {
                float val = fabsf(ptr[i]);
                sum += pow(val, p);
            }
            // 结果 = sum^(1/p)
            Y[n * C + c] = (float)pow(sum, 1.0 / p);
        }
    }
    return 0;
}
