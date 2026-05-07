#include "../kernels.h"
#include "../../../ennf_op_params.h"
#include <math.h>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define USE_AVX2 1
#else
#define USE_AVX2 0
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

/* BatchNormalization */
int op_batch_norm(SpinnTensor **in, int n_in,
                  void *params, uint16_t params_size,
                  SpinnTensor **out, int n_out) {
    if (n_in < 5 || n_out < 1) return -1;
    
    float *X = (float*)in[0]->data;
    float *scale = (float*)in[1]->data;
    float *bias = (float*)in[2]->data;
    float *running_mean = (float*)in[3]->data;
    float *running_var = (float*)in[4]->data;
    float *Y = (float*)out[0]->data;
    
    float eps = 1e-5f;
    if (params && params_size >= sizeof(ENNF_BatchNormParams)) {
        ENNF_BatchNormParams *p = (ENNF_BatchNormParams*)params;
        eps = p->epsilon;
    }
    
    int N = in[0]->dims[0];
    int C = in[0]->dims[1];
    int spatial = 1;
    for (int d = 2; d < in[0]->ndim; d++) spatial *= in[0]->dims[d];
    
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(N * C >= 16)
    #endif
    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            float mean = running_mean[c];
            float var = running_var[c];
            float s = scale[c];
            float b = bias[c];
            float inv_std = 1.0f / sqrtf(var + eps);
            
            float *x_ptr = X + (n * C + c) * spatial;
            float *y_ptr = Y + (n * C + c) * spatial;
            
            /* 预计算系数 */
            float alpha = s * inv_std;
            float beta = b - mean * alpha;
            
            int i = 0;
#if USE_AVX2
            __m256 valpha = _mm256_set1_ps(alpha);
            __m256 vbeta = _mm256_set1_ps(beta);
            
            for (; i + 7 < spatial; i += 8) {
                __m256 vx = _mm256_loadu_ps(x_ptr + i);
                __m256 vy = _mm256_fmadd_ps(vx, valpha, vbeta);
                _mm256_storeu_ps(y_ptr + i, vy);
            }
#endif
            
            for (; i < spatial; i++) {
                y_ptr[i] = x_ptr[i] * alpha + beta;
            }
        }
    }
    return 0;
}
