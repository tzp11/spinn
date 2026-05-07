#include "../kernels.h"
#include <string.h>

/* Slice: 多维张量切片 (支持 axes, steps)
 *
 * in[0]: data
 * in[1]: starts
 * in[2]: ends
 * in[3]: axes  (optional, default 0..ndim-1)
 * in[4]: steps (optional, default all 1)
 */
int op_slice(SpinnTensor **in, int n_in,
             void *params, uint16_t params_size,
             SpinnTensor **out, int n_out) {
    if (n_in < 3 || n_out < 1) return -1;

    const float *X = (const float*)in[0]->data;
    float *Y = (float*)out[0]->data;
    int ndim = in[0]->ndim;
    int num_axes = in[1]->elem_count;

    const int64_t *starts = (const int64_t*)in[1]->data;
    const int64_t *ends   = (const int64_t*)in[2]->data;
    const int64_t *axes   = (n_in >= 4 && in[3] && in[3]->data) ? (const int64_t*)in[3]->data : NULL;
    const int64_t *steps  = (n_in >= 5 && in[4] && in[4]->data) ? (const int64_t*)in[4]->data : NULL;

    /* 解析每个轴的切片参数 */
    int64_t sl_start[8], sl_end[8], sl_step[8], sl_dim_out[8];
    int64_t src_dims[8];
    for (int d = 0; d < ndim; d++) {
        src_dims[d] = in[0]->dims[d];
        sl_start[d] = 0;
        sl_end[d]   = src_dims[d];
        sl_step[d]  = 1;
        sl_dim_out[d] = src_dims[d];
    }

    for (int i = 0; i < num_axes; i++) {
        int ax = axes ? (int)axes[i] : i;
        if (ax < 0) ax += ndim;
        if (ax < 0 || ax >= ndim) return -1;

        int64_t dim = src_dims[ax];
        int64_t s = starts[i];
        int64_t e = ends[i];
        int64_t st = steps ? steps[i] : 1;
        if (st == 0) return -1;

        /* 负索引归一化 */
        if (s < 0) s += dim;
        if (e < 0) e += dim;

        /* clamp + step 方向调整 */
        if (st > 0) {
            if (s < 0) s = 0;
            if (e > dim) e = dim;
            if (s > e) s = e;
        } else {
            if (s >= dim) s = dim - 1;
            if (e < -1) e = -1;
            if (s < e) s = e;
        }

        sl_start[ax] = s;
        sl_end[ax]   = e;
        sl_step[ax]  = st;
        /* 计算输出维度 */
        int64_t out_len = 0;
        if (st > 0) {
            for (int64_t idx = s; idx < e; idx += st) out_len++;
        } else {
            for (int64_t idx = s; idx > e; idx += st) out_len++;
        }
        sl_dim_out[ax] = out_len;
    }

    /* 递归复制: 逐元素遍历输出, 计算源索引 */
    int64_t out_total = 1;
    for (int d = 0; d < ndim; d++) out_total *= sl_dim_out[d];
    if (out_total <= 0) return 0;

    /* 计算源数据的 strides (row-major) */
    int64_t src_stride[8];
    src_stride[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; d--)
        src_stride[d] = src_stride[d + 1] * src_dims[d + 1];

    /* 逐输出元素计算源偏移 */
    int64_t out_idx[8] = {0};
    for (int64_t oi = 0; oi < out_total; oi++) {
        /* 从线性索引计算多维输出索引 */
        int64_t tmp = oi;
        for (int d = ndim - 1; d >= 0; d--) {
            out_idx[d] = tmp % sl_dim_out[d];
            tmp /= sl_dim_out[d];
        }
        /* 计算源偏移 */
        int64_t src_off = 0;
        for (int d = 0; d < ndim; d++) {
            int64_t si = sl_start[d] + out_idx[d] * sl_step[d];
            src_off += si * src_stride[d];
        }
        Y[oi] = X[src_off];
    }

    return 0;
}
