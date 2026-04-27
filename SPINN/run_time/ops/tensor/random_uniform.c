#include "../kernels.h"
#include <stdlib.h>
#include <time.h>

/* RandomUniform: 生成均匀分布随机数 */
int op_random_uniform(SpinnTensor **in, int n_in,
                      void *params, uint16_t params_size,
                      SpinnTensor **out, int n_out) {
    if (n_out < 1) return -1;
    
    float low = 0.0f;
    float high = 1.0f;
    
    static int seeded = 0;
    if (!seeded) { srand(time(NULL)); seeded = 1; }
    
    float *Y = (float*)out[0]->data;
    uint32_t n = out[0]->elem_count;
    
    for (uint32_t i = 0; i < n; i++) {
        Y[i] = low + (high - low) * ((float)rand() / RAND_MAX);
    }
    return 0;
}
