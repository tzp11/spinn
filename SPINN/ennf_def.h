/*
 * ennf_def.h - ENNF (Embedded Neural Network Format) 核心格式定义
 * 
 * 版本: 2.0
 * 特性: 
 *   - params_size 字段实现自描述节点
 *   - 支持 200+ ONNX 算子
 *   - 支持最多 8 维张量
 */

#ifndef __ENNF_DEF_H__
#define __ENNF_DEF_H__

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * 魔数和版本
 * ============================================================ */
#define ENNF_MAGIC          0x454E4E46  /* 'ENNF' */
#define ENNF_VERSION_MAJOR  2
#define ENNF_VERSION_MINOR  0

/* ============================================================
 * 数据类型枚举 (对应 ONNX TensorProto.DataType)
 * ============================================================ */
typedef enum {
    ENNF_TYPE_UNDEFINED = 0,
    ENNF_TYPE_FLOAT32   = 1,    /* float */
    ENNF_TYPE_UINT8     = 2,    /* uint8_t */
    ENNF_TYPE_INT8      = 3,    /* int8_t */
    ENNF_TYPE_UINT16    = 4,    /* uint16_t */
    ENNF_TYPE_INT16     = 5,    /* int16_t */
    ENNF_TYPE_INT32     = 6,    /* int32_t */
    ENNF_TYPE_INT64     = 7,    /* int64_t */
    ENNF_TYPE_STRING    = 8,    /* string (不常用) */
    ENNF_TYPE_BOOL      = 9,    /* bool */
    ENNF_TYPE_FLOAT16   = 10,   /* half float */
    ENNF_TYPE_FLOAT64   = 11,   /* double */
    ENNF_TYPE_UINT32    = 12,   /* uint32_t */
    ENNF_TYPE_UINT64    = 13,   /* uint64_t */
    ENNF_TYPE_COMPLEX64 = 14,   /* complex<float> */
    ENNF_TYPE_COMPLEX128= 15,   /* complex<double> */
    ENNF_TYPE_BFLOAT16  = 16,   /* bfloat16 */
    ENNF_TYPE_FLOAT8E4M3FN = 17,
    ENNF_TYPE_FLOAT8E4M3FNUZ = 18,
    ENNF_TYPE_FLOAT8E5M2 = 19,
    ENNF_TYPE_FLOAT8E5M2FNUZ = 20,
    ENNF_TYPE_UINT4     = 21,
    ENNF_TYPE_INT4      = 22,
} ENNF_DataType;

/* ============================================================
 * AutoPad 枚举 (用于 Conv, Pool 等)
 * ============================================================ */
typedef enum {
    ENNF_AUTOPAD_NOTSET = 0,
    ENNF_AUTOPAD_SAME_UPPER = 1,
    ENNF_AUTOPAD_SAME_LOWER = 2,
    ENNF_AUTOPAD_VALID = 3,
} ENNF_AutoPad;

/* ============================================================
 * 文件头结构 (64 bytes)
 * ============================================================ */
typedef struct {
    /* 魔数与版本 (8 bytes) */
    uint32_t magic;             /* 0x454E4E46 */
    uint16_t version_major;     /* 主版本号 */
    uint16_t version_minor;     /* 次版本号 */
    
    /* 基本统计 (16 bytes) */
    uint32_t num_nodes;         /* 节点数量 */
    uint32_t num_tensors;       /* 张量总数 */
    uint32_t num_weights;       /* 权重张量数 */
    uint32_t total_weight_bytes;/* 权重数据总字节数 */
    
    /* 图信息 (8 bytes) */
    uint16_t num_inputs;        /* 模型输入数 */
    uint16_t num_outputs;       /* 模型输出数 */
    uint16_t opset_version;     /* ONNX Opset 版本 */
    uint16_t flags;             /* 标志位 */
    
    /* 偏移量索引 (32 bytes) */
    uint32_t graph_meta_offset;     /* 图元数据偏移 */
    uint32_t node_table_offset;     /* 节点表偏移 */
    uint32_t tensor_meta_offset;    /* 张量元数据偏移 */
    uint32_t weight_data_offset;    /* 权重数据偏移 */
    uint32_t string_table_offset;   /* 字符串表偏移 (调试用) */
    uint32_t reserved[3];           /* 保留字段 */
} ENNF_Header;

/* Header.flags 位定义 */
#define ENNF_FLAG_QUANTIZED     (1 << 0)    /* 包含量化模型 */
#define ENNF_FLAG_COMPRESSED    (1 << 1)    /* 权重已压缩 */
#define ENNF_FLAG_DYNAMIC_SHAPE (1 << 2)    /* 支持动态形状 */
#define ENNF_FLAG_DEBUG_INFO    (1 << 3)    /* 包含调试信息 */

/* ============================================================
 * 节点基础结构 (8 bytes 固定头 + 变长数据)
 * ============================================================ */
typedef struct {
    uint16_t op_type;           /* 算子类型枚举 (ENNF_OpType) */
    uint8_t  num_inputs;        /* 输入数量 */
    uint8_t  num_outputs;       /* 输出数量 */
    uint16_t params_size;       /* 参数段字节数 (关键: 自描述) */
    uint8_t  flags;             /* 节点标志位 */
    uint8_t  reserved;          /* 对齐到 8 字节 */
    
    /* 后续变长数据 (不在结构体内):
     * uint16_t input_ids[num_inputs];
     * uint16_t output_ids[num_outputs];
     * uint8_t  params[params_size];
     */
} ENNF_NodeBase;

/* Node.flags 位定义 */
#define ENNF_NODE_FLAG_FUSED    (1 << 0)    /* 已与后续算子融合 */
#define ENNF_NODE_FLAG_SKIP     (1 << 1)    /* 跳过执行 (已融合到前节点) */

/* ============================================================
 * 张量元数据结构 (40 bytes)
 * ============================================================ */
#define ENNF_MAX_DIMS 8

typedef struct {
    uint16_t tensor_id;         /* 张量唯一 ID */
    uint8_t  dtype;             /* 数据类型 (ENNF_DataType) */
    uint8_t  ndim;              /* 维度数量 (0-8) */
    uint32_t dims[ENNF_MAX_DIMS]; /* 维度数组 */
    uint32_t elem_count;        /* 元素总数 */
    uint8_t  is_weight;         /* 1=权重(静态), 0=激活值(运行时) */
    uint8_t  ref_count;         /* 引用计数 (用于内存管理) */
    uint8_t  reserved[2];       /* 对齐 */
} ENNF_TensorMeta;

/* ============================================================
 * 权重偏移量索引条目 (16 bytes)
 * ============================================================ */
typedef struct {
    uint16_t tensor_id;         /* 张量 ID */
    uint16_t reserved;
    uint32_t data_offset;       /* 相对 weight_data_offset 的偏移 */
    uint32_t data_size;         /* 数据大小 (字节) */
    uint32_t reserved2;
} ENNF_WeightEntry;

/* ============================================================
 * 图元数据 (变长)
 * ============================================================ */
/* 结构:
 * uint16_t num_inputs;
 * uint16_t input_ids[num_inputs];
 * uint16_t num_outputs;
 * uint16_t output_ids[num_outputs];
 */

/* ============================================================
 * 辅助宏
 * ============================================================ */

/* 获取数据类型的字节大小 */
static inline size_t ennf_dtype_size(ENNF_DataType dtype) {
    switch (dtype) {
    case ENNF_TYPE_FLOAT32:  return 4;
    case ENNF_TYPE_FLOAT64:  return 8;
    case ENNF_TYPE_FLOAT16:  return 2;
    case ENNF_TYPE_BFLOAT16: return 2;
    case ENNF_TYPE_INT8:     return 1;
    case ENNF_TYPE_UINT8:    return 1;
    case ENNF_TYPE_INT16:    return 2;
    case ENNF_TYPE_UINT16:   return 2;
    case ENNF_TYPE_INT32:    return 4;
    case ENNF_TYPE_UINT32:   return 4;
    case ENNF_TYPE_INT64:    return 8;
    case ENNF_TYPE_UINT64:   return 8;
    case ENNF_TYPE_BOOL:     return 1;
    case ENNF_TYPE_INT4:     return 1; /* 实际是 4-bit, 按 1 byte 处理 */
    case ENNF_TYPE_UINT4:    return 1;
    default:                 return 0;
    }
}

#ifdef __cplusplus
}
#endif

#endif /* __ENNF_DEF_H__ */
