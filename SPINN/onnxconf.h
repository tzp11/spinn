/* 
 * ONNX 配置头文件（最小版本）
 * 只包含 protobuf-c 需要的基本定义
 */

#ifndef __ONNXCONF_H__
#define __ONNXCONF_H__

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <assert.h>

/* protobuf-c 使用的包装函数，映射到标准 C 库函数 */
#define onnx_malloc     malloc
#define onnx_free       free
#define onnx_memcpy     memcpy
#define onnx_memset     memset
#define onnx_memmove    memmove
#define onnx_strcmp     strcmp
#define onnx_strlen     strlen
#define onnx_assert     assert

#endif /* __ONNXCONF_H__ */
