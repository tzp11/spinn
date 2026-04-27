/*
 * spinn_ops.h - 算子库接口
 */

#ifndef __SPINN_OPS_H__
#define __SPINN_OPS_H__

#include "spinn_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子调度器
 * @param op_type 算子类型 (ENNF_OpType)
 * @param inputs  输入 Tensor 指针数组
 * @param n_in    输入数量
 * @param params  算子参数
 * @param params_size 参数字节数
 * @param outputs 输出 Tensor 指针数组
 * @param n_out   输出数量
 * @return 0 成功, <0 失败
 */
int spinn_dispatch_op(uint16_t op_type,
                      SpinnTensor **inputs, int n_in,
                      void *params, uint16_t params_size,
                      SpinnTensor **outputs, int n_out);

/* 单个算子 Kernel 函数类型 */
typedef int (*SpinnOpKernel)(SpinnTensor **in, int n_in,
                             void *params, uint16_t params_size,
                             SpinnTensor **out, int n_out);

#ifdef __cplusplus
}
#endif

#endif /* __SPINN_OPS_H__ */
