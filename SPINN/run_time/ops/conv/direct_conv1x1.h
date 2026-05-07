#ifndef __DIRECT_CONV1X1_H__
#define __DIRECT_CONV1X1_H__

/*
 * Direct Conv 1×1 (无 im2col)
 * 
 * 适用条件: KH=KW=1, SH=SW=1, PH=PW=0
 */

void direct_conv1x1(
    const float *X,     // [C, H, W]
    const float *W,     // [OC, C]
    float *Y,           // [OC, H, W]
    const float *bias,  // [OC] or NULL
    int C, int H, int Wdim, int OC
);

void direct_conv1x1_fused(
    const float *X,     // [C, H, W]
    const float *W,     // [OC, C]
    float *Y,           // [OC, H, W]
    const float *bias,  // [OC] or NULL
    int C, int H, int Wdim, int OC,
    int fused_act       // 0=None, 1=ReLU, 3=SiLU
);

#endif
