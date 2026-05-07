#ifndef NCHWC_H
#define NCHWC_H

#include <stddef.h>

/* NCHWc block size for AVX2 */
#define NCHWC_BLOCK_SIZE 8

/* NCHW -> NCHWc 转换 */
void nchwc_reorder_input(
    const float *src,      /* NCHW: [C][H][W] */
    float *dst,            /* NCHWc: [C/8][H][W][8] */
    int C, int H, int W
);

/* NCHWc -> NCHW 转换 */
void nchwc_reorder_output(
    const float *src,      /* NCHWc: [C/8][H][W][8] */
    float *dst,            /* NCHW: [C][H][W] */
    int C, int H, int W
);

/* NCHWc 卷积 (1×1, stride=1) */
void nchwc_conv1x1(
    const float *input,    /* NCHWc: [IC/8][H][W][8] */
    const float *weight,   /* NCHWc: [OC/8][IC/8][8][8] */
    const float *bias,
    float *output,         /* NCHWc: [OC/8][H][W][8] */
    int IC, int OC, int H, int W
);

#endif
