#include "../kernels.h"

/* Expand: 按照shape广播张量 */
int op_expand(SpinnTensor **in, int n_in,
              void *params, uint16_t params_size,
              SpinnTensor **out, int n_out) {
    if (n_in < 2 || n_out < 1) return -1;
    
    // in[0]: data, in[1]: shape
    float *X = (float*)in[0]->data;
    float *Y = (float*)out[0]->data;
    
    int ndim_in = in[0]->ndim;
    int ndim_out = out[0]->ndim;
    
    uint32_t n = out[0]->elem_count;
    
    // 计算输入的stride
    uint32_t in_strides[8];
    in_strides[ndim_in - 1] = 1;
    for (int i = ndim_in - 2; i >= 0; i--) {
        in_strides[i] = in_strides[i + 1] * in[0]->dims[i + 1];
    }
    
    // 计算输出的stride
    uint32_t out_strides[8];
    out_strides[ndim_out - 1] = 1;
    for (int i = ndim_out - 2; i >= 0; i--) {
        out_strides[i] = out_strides[i + 1] * out[0]->dims[i + 1];
    }
    
    // 遍历输出的每个元素
    for (uint32_t idx = 0; idx < n; idx++) {
        uint32_t coords[8];
        uint32_t tmp = idx;
        
        // 计算输出坐标
        for (int d = 0; d < ndim_out; d++) {
            coords[d] = tmp / out_strides[d];
            tmp = tmp % out_strides[d];
        }
        
        // 计算输入索引（考虑广播）
        uint32_t in_idx = 0;
        int offset = ndim_out - ndim_in;
        for (int d = 0; d < ndim_in; d++) {
            int out_d = d + offset;
            uint32_t coord = (out_d >= 0) ? coords[out_d] : 0;
            // 如果输入维度是1，使用0索引（广播）
            if (in[0]->dims[d] == 1) coord = 0;
            in_idx += coord * in_strides[d];
        }
        
        Y[idx] = X[in_idx];
    }
    
    return 0;
}
