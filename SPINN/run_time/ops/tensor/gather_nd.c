#include "../kernels.h"

/* GatherND: N维索引聚集 */
int op_gather_nd(SpinnTensor **in, int n_in,
                 void *params, uint16_t params_size,
                 SpinnTensor **out, int n_out) {
    (void)params; (void)params_size;
    if (n_in < 2 || n_out < 1) return -1;
    
    float *data = (float*)in[0]->data;
    int64_t *indices = (int64_t*)in[1]->data;
    float *Y = (float*)out[0]->data;
    
    int data_ndim = in[0]->ndim;
    int indices_ndim = in[1]->ndim;
    int indices_last_dim = in[1]->dims[indices_ndim - 1];
    
    // 计算data的stride
    uint32_t data_strides[8];
    data_strides[data_ndim - 1] = 1;
    for (int i = data_ndim - 2; i >= 0; i--) {
        data_strides[i] = data_strides[i + 1] * in[0]->dims[i + 1];
    }
    
    // 计算indices的批次大小
    uint32_t num_batches = 1;
    for (int i = 0; i < indices_ndim - 1; i++) {
        num_batches *= in[1]->dims[i];
    }
    
    // 每个批次输出的大小
    uint32_t output_slice_size = 1;
    for (int i = indices_last_dim; i < data_ndim; i++) {
        output_slice_size *= in[0]->dims[i];
    }
    
    for (uint32_t batch = 0; batch < num_batches; batch++) {
        // 计算data索引
        uint32_t data_idx = 0;
        for (int i = 0; i < indices_last_dim; i++) {
            int64_t idx = indices[batch * indices_last_dim + i];
            if (idx < 0) idx += in[0]->dims[i];
            data_idx += idx * data_strides[i];
        }
        
        // 复制切片
        for (uint32_t j = 0; j < output_slice_size; j++) {
            Y[batch * output_slice_size + j] = data[data_idx + j];
        }
    }
    
    return 0;
}
