#include "../kernels.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>

/* RandomNormal: 生成正态分布随机数 */
int op_random_normal(SpinnTensor **in, int n_in,
                     void *params, uint16_t params_size,
                     SpinnTensor **out, int n_out) {
    if (n_out < 1) return -1;
    
    float mean = 0.0f;
    float scale = 1.0f;
    
    // 简化：使用 Box-Muller 变换生成正态分布
    static int seeded = 0;
    if (!seeded) { srand(time(NULL)); seeded = 1; }
    
    float *Y = (float*)out[0]->data;
    uint32_t n = out[0]->elem_count;
    
    for (uint32_t i = 0; i < n; i += 2) {
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
        float z1 = sqrtf(-2.0f * logf(u1)) * sinf(2.0f * 3.14159265f * u2);
        Y[i] = mean + scale * z0;
        if (i + 1 < n) Y[i + 1] = mean + scale * z1;
    }
    return 0;
}
