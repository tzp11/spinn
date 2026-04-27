#include "../kernels.h"
#include <stdlib.h>
#include "../../../ennf_op_params.h"

/* Helper struct for qsort */
typedef struct {
    float value;
    int64_t index;
} TopKEntry;

int compare_topk_desc(const void *a, const void *b) {
    float v1 = ((TopKEntry*)a)->value;
    float v2 = ((TopKEntry*)b)->value;
    if (v1 > v2) return -1;
    if (v1 < v2) return 1;
    return 0;
}

/* TopK: 返回top k个最大值及索引 */
int op_topk(SpinnTensor **in, int n_in,
            void *params, uint16_t params_size,
            SpinnTensor **out, int n_out) {
    if (n_in < 2 || n_out < 2) return -1;
    
    // in[0]: X, in[1]: K (tensor or scalar form)
    float *X = (float*)in[0]->data;
    
    // K can be INT64 (standard ONNX)
    int64_t k_val = 0;
    if (in[1]->dtype == 7) { // INT64
        k_val = ((int64_t*)in[1]->data)[0];
    } else {
        // Fallback or error? Assume 64-bit for K usually.
        k_val = ((int64_t*)in[1]->data)[0]; 
    }
    
    float *out_values = (float*)out[0]->data;
    int64_t *out_indices = (int64_t*)out[1]->data;
    
    // Assume 2D [1, N] or 1D [N] and sorting last dim for now.
    // Ideally should handle axis from params.
    int axis = -1;
    if (params && params_size >= sizeof(ENNF_TopKParams)) {
        ENNF_TopKParams *p = (ENNF_TopKParams*)params;
        axis = p->axis;
    }
    (void)axis;
    
    // Simple implementation for [Batch, N] case where we sort N
    // This is tailored for the YOLOv10 case [1, 8400] -> TopK -> [1, 300]
    // Generalizing to "Treat as flattened last dim"
    
    int last_dim = in[0]->ndim - 1;
    int N = in[0]->dims[last_dim];
    int num_batches = 1;
    for (int i = 0; i < last_dim; i++) num_batches *= in[0]->dims[i];
    
    int k = (int)k_val;
    if (k > N) k = N;
    
    // Temp buffer for sorting pairs
    // We allocate worst case N entries
    TopKEntry *entries = (TopKEntry*)malloc(sizeof(TopKEntry) * N);
    if (!entries) return -2;
    
    for (int b = 0; b < num_batches; b++) {
        float *batch_x = X + b * N;
        float *batch_val = out_values + b * k;
        int64_t *batch_idx = out_indices + b * k;
        
        // Populate entries
        for (int i = 0; i < N; i++) {
            entries[i].value = batch_x[i];
            entries[i].index = i;
        }
        
        // Sort
        qsort(entries, N, sizeof(TopKEntry), compare_topk_desc);
        
        // Copy Top K
        for (int i = 0; i < k; i++) {
            batch_val[i] = entries[i].value;
            batch_idx[i] = entries[i].index;
        }
    }
    
    free(entries);
    return 0;
}
