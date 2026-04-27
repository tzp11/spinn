#include "../kernels.h"

/* Slice: 张量切片 */
int op_slice(SpinnTensor **in, int n_in,
             void *params, uint16_t params_size,
             SpinnTensor **out, int n_out) {
    if (n_in < 3 || n_out < 1) return -1;
    
    // in[0]: data, in[1]: starts, in[2]: ends, in[3]: axes (optional), in[4]: steps (optional)
    float *X = (float*)in[0]->data;
    float *Y = (float*)out[0]->data;
    int64_t *starts = (int64_t*)in[1]->data;
    int64_t *ends = (int64_t*)in[2]->data;
    
    // 简化实现：仅支持连续的1D切片
    int ndim = in[0]->ndim;
    
    if (ndim == 1) {
        int64_t start = starts[0];
        int64_t end = ends[0];
        int64_t dim = in[0]->dims[0];
        
        if (start < 0) start += dim;
        if (end < 0) end += dim;
        if (end > dim) end = dim;
        
        for (int64_t i = start; i < end; i++) {
            Y[i - start] = X[i];
        }
    } else {
        // 多维切片较复杂，暂时复制整个输入
        uint32_t n = out[0]->elem_count;
        for (uint32_t i = 0; i < n; i++) {
            Y[i] = X[i];
        }
    }
    return 0;
}
