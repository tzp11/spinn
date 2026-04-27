#include "../kernels.h"
#include <math.h>
#include <stdio.h>

/* Div: out = in[0] / in[1] */
int op_div(SpinnTensor **in, int n_in,
           void *params, uint16_t params_size,
           SpinnTensor **out, int n_out) {
    (void)params; (void)params_size;
    if (n_in < 2 || n_out < 1) return -1;
    if (!in[0]->data || !in[1]->data || !out[0]->data) {
        printf("Error: Div input/output data is NULL\n");
        return -2;
    }
    
    int dtype = in[0]->dtype; 
    
    uint32_t n = out[0]->elem_count;
    uint32_t n_a = in[0]->elem_count;
    uint32_t n_b = in[1]->elem_count;
    if (n_a == 0 || n_b == 0) return 0;
    
    if (dtype == 1) { // FLOAT
        float *a = (float*)in[0]->data;
        float *b = (float*)in[1]->data;
        float *y = (float*)out[0]->data;
        float eps = 1e-6f;
        for (uint32_t i = 0; i < n; i++) {
            float val_b = b[i % n_b];
            if (fabsf(val_b) < eps) val_b = eps;
            y[i] = a[i % n_a] / val_b;
        }
    } else if (dtype == 7) { // INT64
        int64_t *a = (int64_t*)in[0]->data;
        int64_t *b = (int64_t*)in[1]->data;
        int64_t *y = (int64_t*)out[0]->data;
        for (uint32_t i = 0; i < n; i++) {
            int64_t vb = b[i % n_b];
            y[i] = (vb != 0) ? (a[i % n_a] / vb) : 0;
        }
    } else if (dtype == 6) { // INT32
        int32_t *a = (int32_t*)in[0]->data;
        int32_t *b = (int32_t*)in[1]->data;
        int32_t *y = (int32_t*)out[0]->data;
        for (uint32_t i = 0; i < n; i++) {
            int32_t vb = b[i % n_b];
            y[i] = (vb != 0) ? (a[i % n_a] / vb) : 0;
        }
    } else {
        return -2;
    }
    return 0;
}
