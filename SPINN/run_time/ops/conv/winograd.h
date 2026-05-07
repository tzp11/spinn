#ifndef __WINOGRAD_H__
#define __WINOGRAD_H__

/*
 * Winograd F(2,3) 3×3 卷积接口
 *
 * 调用条件: KH=KW=3, SH=SW=1, group=1
 */

/* 缓存权重变换: weight[OC*C*9] → U[16*OC*C], 返回需 free() 的缓冲 */
float* winograd_pack_weight(const float *weight, int OC, int C);

int winograd_conv_3x3(const float *X, int N, int C, int H, int W,
                      const float *U_packed, int OC,
                      const float *bias,
                      int PH, int PW,
                      float *Y, int OH, int OW,
                      int fused_relu);

#endif
