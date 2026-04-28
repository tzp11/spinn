#include "../kernels.h"
#include "../../../ennf_op_params.h"
#include <string.h>

/*
 * Transpose 优化:
 *   - 嵌套循环遍历输出张量, 每层用计数器递增, 完全消除 div/mod
 *   - 检测尾部"恒等"维度 (perm[d]==d 的最长后缀), 内层 memcpy
 *   - 退化情况 (perm 是恒等映射) 走 memcpy 快路径
 *
 * 输出顺序遍历: out_idx 是连续递增的, 减少写入侧 cache miss.
 * 对每个输出位置反查输入位置: in_idx = sum(out_coords[d] * in_stride[perm[d]])
 *   每层维度的 in_stride[perm[d]] 提前预计算成 out_stride_in_input[d]
 */
int op_transpose(SpinnTensor **in, int n_in,
                 void *params, uint16_t params_size,
                 SpinnTensor **out, int n_out) {
    (void)n_in; (void)n_out;
    if (n_in < 1 || n_out < 1) return -1;

    float *src = (float*)in[0]->data;
    float *dst = (float*)out[0]->data;
    int ndim = in[0]->ndim;
    uint32_t total = in[0]->elem_count;
    if (total == 0) return 0;

    /* perm[i] = 输入张量的哪个维度作为输出第 i 维 */
    uint8_t perm[ENNF_MAX_DIMS];
    if (params && params_size >= sizeof(ENNF_TransposeParams)) {
        ENNF_TransposeParams *p = (ENNF_TransposeParams*)params;
        for (int i = 0; i < ndim; i++) perm[i] = p->perm[i];
    } else {
        /* 默认: 逆序 */
        for (int i = 0; i < ndim; i++) perm[i] = (uint8_t)(ndim - 1 - i);
    }

    /* 输入 strides (元素单位) */
    uint32_t in_strides[ENNF_MAX_DIMS];
    in_strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        in_strides[i] = in_strides[i + 1] * in[0]->dims[i + 1];
    }

    /* 恒等 perm: 直接 memcpy */
    int identity = 1;
    for (int i = 0; i < ndim; i++) {
        if (perm[i] != i) { identity = 0; break; }
    }
    if (identity) {
        memcpy(dst, src, total * sizeof(float));
        return 0;
    }

    /* 输出维度 = 输入维度按 perm 重排 */
    uint32_t out_dims[ENNF_MAX_DIMS];
    for (int i = 0; i < ndim; i++) out_dims[i] = in[0]->dims[perm[i]];

    /* 对每个输出维度 d, 输出第 d 维步进 1 时, 输入指针偏移 in_strides[perm[d]] 个元素 */
    uint32_t step[ENNF_MAX_DIMS];
    for (int i = 0; i < ndim; i++) step[i] = in_strides[perm[i]];

    /* 检测尾部"恒等"后缀: 最大 contig 使得 perm[ndim-contig..ndim-1] = ndim-contig..ndim-1
     * 这意味着最后 contig 个维度顺序未变, 且步进 step[d]=in_strides[d], 内层连续 */
    int contig = 0;
    while (contig < ndim && perm[ndim - 1 - contig] == ndim - 1 - contig) {
        contig++;
    }
    /* 内层连续 copy 长度 (元素数) */
    uint32_t inner_len = 1;
    for (int i = ndim - contig; i < ndim; i++) inner_len *= out_dims[i];

    int outer_ndim = ndim - contig;  /* 需要嵌套循环的维度数 */

    /* 全部恒等已在前面处理, outer_ndim >= 1 */

    /* 计数器和当前输入偏移 */
    uint32_t coord[ENNF_MAX_DIMS] = {0};
    uint32_t in_off = 0;
    float *out_p = dst;
    uint32_t out_remaining = total;  /* 仅用于安全裕度 */
    (void)out_remaining;

    /*
     * 嵌套循环展开为 while: 每次处理 inner_len 个连续元素
     * 然后递增计数器 (从最里层 outer 维度开始向上进位)
     */
    while (1) {
        /* 内层连续 copy: src[in_off .. in_off+inner_len) -> dst[out_p .. out_p+inner_len) */
        if (inner_len == 1) {
            *out_p = src[in_off];
        } else {
            memcpy(out_p, src + in_off, inner_len * sizeof(float));
        }
        out_p += inner_len;

        /* 递增计数器 (从 outer_ndim-1 向 0 进位) */
        int d = outer_ndim - 1;
        while (d >= 0) {
            coord[d]++;
            in_off += step[d];
            if (coord[d] < out_dims[d]) break;
            /* 该维度满了, 回退 */
            in_off -= (uint32_t)out_dims[d] * step[d];
            coord[d] = 0;
            d--;
        }
        if (d < 0) break;  /* 所有维度循环完毕 */
    }

    return 0;
}
