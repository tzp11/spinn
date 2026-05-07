#ifndef DIRECT_CONV3X3_H
#define DIRECT_CONV3X3_H

void direct_conv3x3_s1(
    const float *X, int C, int H, int W,
    const float *weight, int OC,
    const float *bias,
    int PH, int PW,
    float *Y, int OH, int OW
);

#endif
