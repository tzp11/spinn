/*
 * onnx2ennf.c - 完整 ENNF 转换器 (Version 2.0)
 * 
 * 功能：
 * 1. 解析 ONNX Protobuf
 * 2. 映射 Tensor ID 并计算引用计数
 * 3. 提取算子参数 (支持 params_size 自描述)
 * 4. 生成 ENNF 二进制文件
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string.h>
#include <libgen.h> // for dirname
#include "onnx.proto3.pb-c.h"
#include "ennf_def.h"
#include "ennf_op_types.h"
#include "ennf_op_params.h"
#include "graph_opt.h"

// 外部声明：算子参数提取函数 (在 onnx2ennf_ops.c 中实现)
void extract_op_params(Onnx__NodeProto *node, ENNF_OpType type, 
                       uint8_t *buf, uint16_t *size);

#define MAX_TENSORS 10000

typedef struct {
    char *name;
    int id;
    int is_weight;
    Onnx__TensorProto *initializer;
    Onnx__ValueInfoProto *value_info;
    Onnx__TypeProto *type_proto; // 补充：有些tensor只有type信息
    int ref_count;
} TensorEntry;

TensorEntry tensors[MAX_TENSORS];
int tensor_count = 0;

// 查找或创建 Tensor ID
int get_tensor_id(const char *name) {
    if (!name) return -1;
    for (int i = 0; i < tensor_count; i++) {
        if (strcmp(tensors[i].name, name) == 0) {
            return tensors[i].id;
        }
    }
    // New Tensor
    if (tensor_count >= MAX_TENSORS) {
        fprintf(stderr, "Error: Max tensors limit reached (%d)\n", MAX_TENSORS);
        exit(1);
    }
    tensors[tensor_count].name = strdup(name);
    tensors[tensor_count].id = tensor_count;
    tensors[tensor_count].is_weight = 0;
    tensors[tensor_count].initializer = NULL;
    tensors[tensor_count].value_info = NULL;
    tensors[tensor_count].type_proto = NULL;
    tensors[tensor_count].ref_count = 0;
    tensor_count++;
    return tensor_count - 1;
}

// 增加引用计数
void inc_ref_count(int id) {
    if (id >= 0 && id < tensor_count) {
        tensors[id].ref_count++;
    }
}

// 映射 ONNX DataType 到 ENNF DataType
ENNF_DataType map_dtype(int onnx_dtype) {
    switch (onnx_dtype) {
        case 1:  return ENNF_TYPE_FLOAT32;
        case 2:  return ENNF_TYPE_UINT8;
        case 3:  return ENNF_TYPE_INT8;
        case 4:  return ENNF_TYPE_UINT16;
        case 5:  return ENNF_TYPE_INT16;
        case 6:  return ENNF_TYPE_INT32;
        case 7:  return ENNF_TYPE_INT64;
        case 9:  return ENNF_TYPE_BOOL;
        case 10: return ENNF_TYPE_FLOAT16;
        case 11: return ENNF_TYPE_FLOAT64;
        case 12: return ENNF_TYPE_UINT32;
        case 13: return ENNF_TYPE_UINT64;
        default: return ENNF_TYPE_UNDEFINED;
    }
}


// Helper: Read external data
// Returns malloc'ed buffer (caller must free), or NULL on error
uint8_t* read_external_data(Onnx__TensorProto *proto, const char *onnx_path, size_t *out_len) {
    const char *location = NULL;
    long long offset = 0;
    long long length = -1;
    
    for (size_t i = 0; i < proto->n_external_data; i++) {
        const char *key = proto->external_data[i]->key;
        const char *val = proto->external_data[i]->value;
        if (strcmp(key, "location") == 0) location = val;
        else if (strcmp(key, "offset") == 0) offset = atoll(val);
        else if (strcmp(key, "length") == 0) length = atoll(val);
    }
    
    if (!location) return NULL;
    
    // Construct full path relative to onnx_path
    char *path_copy = strdup(onnx_path);
    char *dir = dirname(path_copy);
    char full_path[1024];
    snprintf(full_path, sizeof(full_path), "%s/%s", dir, location);
    free(path_copy);
    
    FILE *fp = fopen(full_path, "rb");
    if (!fp) {
        // Try current dir as fallback
        fp = fopen(location, "rb");
    }
    if (!fp) {
        fprintf(stderr, "Error: Cannot open external data file: %s\n", full_path);
        return NULL;
    }
    
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    
    if (length < 0) {
        length = fsize - offset; // Read until end if length not specified
    }
    
    if (offset + length > fsize) {
        fprintf(stderr, "Error: External data read out of bounds. File: %ld, Req: %lld + %lld\n", fsize, offset, length);
        fclose(fp);
        return NULL;
    }
    
    fseek(fp, offset, SEEK_SET);
    uint8_t *buf = malloc(length);
    if (!buf) {
        fclose(fp);
        return NULL;
    }
    
    if (fread(buf, 1, length, fp) != length) {
        fprintf(stderr, "Error: Short read from external data\n");
        free(buf);
        fclose(fp);
        return NULL;
    }
    
    fclose(fp);
    *out_len = (size_t)length;
    return buf;
}
int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <input.onnx> <output.ennf>\n", argv[0]);
        return 1;
    }
    const char *onnx_file = argv[1];
    const char *ennf_file = argv[2];

    // 1. Load ONNX
    FILE *fp = fopen(onnx_file, "rb");
    if (!fp) { perror("fopen onnx"); return 1; }
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    // Allocate extra for pad
    uint8_t *buf = malloc(fsize + 8); 
    if (!buf) { perror("malloc"); return 1; }
    size_t n_read = fread(buf, 1, fsize, fp);
    if (n_read != fsize) { fprintf(stderr, "Short read\n"); return 1; }
    fclose(fp);

    Onnx__ModelProto *model = onnx__model_proto__unpack(NULL, fsize, buf);
    if (!model) { fprintf(stderr, "Failed to unpack onnx\n"); return 1; }
    Onnx__GraphProto *graph = model->graph;
    printf("ONNX Loaded. Nodes: %zu, Inputs: %zu, Outputs: %zu\n", 
           graph->n_node, graph->n_input, graph->n_output);

    // 2. Scan Tensors & Build Map
    
    // 2.1 Initializers (Weights)
    int num_weights = 0;
    uint32_t total_weight_bytes = 0;
    for (int i = 0; i < graph->n_initializer; i++) {
        int id = get_tensor_id(graph->initializer[i]->name);
        tensors[id].is_weight = 1;
        tensors[id].initializer = graph->initializer[i];
        num_weights++;
        
        // Check raw_data first
        if (graph->initializer[i]->raw_data.len > 0) {
            total_weight_bytes += graph->initializer[i]->raw_data.len;
        } else if (graph->initializer[i]->n_float_data > 0) {
            total_weight_bytes += graph->initializer[i]->n_float_data * sizeof(float);
        } else if (graph->initializer[i]->n_int64_data > 0) {
            total_weight_bytes += graph->initializer[i]->n_int64_data * sizeof(int64_t);
        } else if (graph->initializer[i]->n_int32_data > 0) {
            total_weight_bytes += graph->initializer[i]->n_int32_data * sizeof(int32_t);
        } else if (graph->initializer[i]->n_external_data > 0) {
             // We won't pre-calculate size accurately here without parsing keys, 
             // but we can trust the 'length' key if present, or just accumulate 0 now 
             // and let the file write update total later.
             // Actually Header.total_weight_bytes relies on this?
             // No, code at Line 400 overwrites header.total_weight_bytes. So this variable is mostly for debug print?
             // Yes it is unused except implicitly. Let's ignore precise count here.
        }
    }

    // 2.2 Inputs
    for (int i = 0; i < graph->n_input; i++) {
        int id = get_tensor_id(graph->input[i]->name);
        tensors[id].value_info = graph->input[i];
    }

    // 2.3 Outputs
    for (int i = 0; i < graph->n_output; i++) {
        int id = get_tensor_id(graph->output[i]->name);
        tensors[id].value_info = graph->output[i];
    }
    
    // 2.4 ValueInfos (Intermediates)
    for (int i = 0; i < graph->n_value_info; i++) {
        int id = get_tensor_id(graph->value_info[i]->name);
        tensors[id].value_info = graph->value_info[i];
    }
    
    // 2.5 Scan Nodes to find implicitly defined tensors & RefCounts
    for (int i = 0; i < graph->n_node; i++) {
        Onnx__NodeProto *node = graph->node[i];
        
        // Scan Outputs first (define them)
        for (int j = 0; j < node->n_output; j++) {
            get_tensor_id(node->output[j]);
        }
        
        // Scan Inputs (reference them -> inc ref_count)
        for (int j = 0; j < node->n_input; j++) {
            int id = get_tensor_id(node->input[j]);
            inc_ref_count(id);
        }
    }
    printf("Total Tensors: %d, Weights: %d\n", tensor_count, num_weights);

    // 2.6 Graph Optimization
    GraphOptContext *opt_ctx = graph_opt_init(graph);
    if (opt_ctx) {
        graph_opt_run(opt_ctx);
    }

    // 3. Start Writing ENNF
    FILE *fout = fopen(ennf_file, "wb");
    if (!fout) { perror("fopen ennf"); return 1; }
    
    // 3.1 Header Placeholder
    ENNF_Header header;
    memset(&header, 0, sizeof(header));
    header.magic = ENNF_MAGIC;
    header.version_major = ENNF_VERSION_MAJOR;
    header.version_minor = ENNF_VERSION_MINOR;
    /* 计算优化后的有效节点数 */
    int effective_nodes = graph->n_node;
    if (opt_ctx) {
        for (int i = 0; i < (int)graph->n_node; i++) {
            if (opt_ctx->node_flags[i] & NODE_FLAG_SKIP) effective_nodes--;
        }
    }
    header.num_nodes = effective_nodes;
    header.num_tensors = tensor_count;
    header.num_weights = num_weights;
    header.total_weight_bytes = 0; // Will update later
    header.num_inputs = graph->n_input;
    header.num_outputs = graph->n_output;
    header.opset_version = (model->n_opset_import > 0) ? model->opset_import[0]->version : 0;
    
    fwrite(&header, sizeof(header), 1, fout);
    
    // 3.2 Offset Table Placeholder (Dense Table: one entry per tensor)
    // Writing placeholder for `num_tensors` entries.
    long offset_table_pos = ftell(fout);
    header.reserved[0] = offset_table_pos; 
    header.weight_data_offset = 0; 
    
    ENNF_WeightEntry dummy_entry = {0};
    for (int i = 0; i < tensor_count; i++) {
        fwrite(&dummy_entry, sizeof(dummy_entry), 1, fout);
    }
    
    // 3.3 Graph Meta
    header.graph_meta_offset = (uint32_t)ftell(fout);
    {
        uint16_t n_in = graph->n_input;
        fwrite(&n_in, 2, 1, fout);
        for (int i=0; i<n_in; i++) {
            uint16_t id = get_tensor_id(graph->input[i]->name);
            fwrite(&id, 2, 1, fout);
        }
        
        uint16_t n_out = graph->n_output;
        fwrite(&n_out, 2, 1, fout);
        for (int i=0; i<n_out; i++) {
            uint16_t id = get_tensor_id(graph->output[i]->name);
            fwrite(&id, 2, 1, fout);
        }
    }
    
    // 3.4 Node Table
    header.node_table_offset = (uint32_t)ftell(fout);
    uint8_t params_buf[1024]; // Temp buffer for params
    
    for (int i = 0; i < (int)graph->n_node; i++) {
        /* 跳过已被融合的节点 */
        if (opt_ctx && (opt_ctx->node_flags[i] & NODE_FLAG_SKIP)) continue;
        
        Onnx__NodeProto *node = graph->node[i];
        
        ENNF_NodeBase base;
        base.op_type = ennf_get_op_type(node->op_type);
        if (base.op_type == OP_UNKNOWN) {
            fprintf(stderr, "Warning: Unknown op type '%s'\n", node->op_type);
        }
        
        base.num_inputs = node->n_input;
        base.num_outputs = node->n_output;
        base.flags = 0;
        base.reserved = 0;
        
        /* 标记融合激活 */
        if (opt_ctx && (opt_ctx->node_flags[i] & NODE_FLAG_HAS_ACT)) {
            base.flags |= ENNF_NODE_FLAG_FUSED;
        }
        
        // Extract Params
        uint16_t p_size = 0;
        extract_op_params(node, base.op_type, params_buf, &p_size);
        
        /* 把融合的激活类型写入 Conv 参数的 reserved 字段 */
        if (opt_ctx && opt_ctx->fused_act[i] != FUSED_ACT_NONE) {
            if (base.op_type == OP_Conv && p_size >= sizeof(ENNF_ConvParams)) {
                ENNF_ConvParams *cp = (ENNF_ConvParams *)params_buf;
                cp->reserved = opt_ctx->fused_act[i];
            }
        }
        
        base.params_size = p_size;
        
        // Write Base
        fwrite(&base, sizeof(base), 1, fout);
        
        // Write Input IDs
        for (int j=0; j<(int)node->n_input; j++) {
            uint16_t id = get_tensor_id(node->input[j]);
            fwrite(&id, 2, 1, fout);
        }
        
        // Write Output IDs
        for (int j=0; j<(int)node->n_output; j++) {
            uint16_t id = get_tensor_id(node->output[j]);
            fwrite(&id, 2, 1, fout);
        }
        
        // Write Params
        if (p_size > 0) {
            fwrite(params_buf, p_size, 1, fout);
        }
    }
    
    // 3.5 Tensor Meta Table
    header.tensor_meta_offset = (uint32_t)ftell(fout);
    for (int i = 0; i < tensor_count; i++) {
        TensorEntry *t = &tensors[i];
        ENNF_TensorMeta meta;
        memset(&meta, 0, sizeof(meta));
        
        meta.tensor_id = t->id;
        meta.ref_count = t->ref_count;
        meta.is_weight = t->is_weight;
        
        // Extract Shape & Type
        int64_t *dims = NULL;
        size_t n_dims = 0;
        int onnx_dtype = 0;
        
        if (t->initializer) {
            dims = t->initializer->dims;
            n_dims = t->initializer->n_dims;
            onnx_dtype = t->initializer->data_type;
        } else if (t->value_info) {
            // Need to dig into ValueInfo -> Type -> TensorType -> Shape
            Onnx__TypeProto *tp = t->value_info->type;
            if (tp && tp->value_case == ONNX__TYPE_PROTO__VALUE_TENSOR_TYPE && tp->tensor_type) {
                onnx_dtype = tp->tensor_type->elem_type;
                if (tp->tensor_type->shape) {
                    n_dims = tp->tensor_type->shape->n_dim;
                }
            }
        }
        
        meta.dtype = map_dtype(onnx_dtype);
        meta.ndim = (n_dims > ENNF_MAX_DIMS) ? ENNF_MAX_DIMS : n_dims;
        
        for (int d = 0; d < meta.ndim; d++) {
            int64_t val = 0;
            if (t->initializer) {
                val = dims[d];
            } else if (t->value_info) {
                // Dig again
                Onnx__TypeProto *tp = t->value_info->type;
                if (tp && tp->tensor_type && tp->tensor_type->shape && tp->tensor_type->shape->dim) {
                    Onnx__TensorShapeProto__Dimension *dim = tp->tensor_type->shape->dim[d];
                    if (dim->value_case == ONNX__TENSOR_SHAPE_PROTO__DIMENSION__VALUE_DIM_VALUE) {
                        val = dim->dim_value;
                    } else {
                        val = 0; // Dynamic
                        header.flags |= ENNF_FLAG_DYNAMIC_SHAPE;
                    }
                }
            }
            meta.dims[d] = (uint32_t)val;
        }
        meta.elem_count = 1;
        for (int d=0; d<meta.ndim; d++) meta.elem_count *= meta.dims[d];
        if (meta.elem_count == 0 && meta.ndim > 0) meta.elem_count = 0; // Contains dynamic dim
        
        fwrite(&meta, sizeof(meta), 1, fout);
    }
    
    // 3.6 Weight Data
    // Align to 16 bytes for SIMD friendliness
    long current_pos = ftell(fout);
    long aligned_pos = (current_pos + 15) & ~15;
    while (current_pos < aligned_pos) {
        fputc(0, fout);
        current_pos++;
    }
    
    header.weight_data_offset = (uint32_t)ftell(fout);
    uint32_t weight_relative_offset = 0;
    
    // Prepare Offset Entries (Dense Array)
    ENNF_WeightEntry *weight_entries = calloc(tensor_count, sizeof(ENNF_WeightEntry));
    
    for (int i = 0; i < tensor_count; i++) {
        TensorEntry *t = &tensors[i];
        
        // Default: 0
        weight_entries[i].tensor_id = t->id;
        weight_entries[i].reserved = 0;
        weight_entries[i].reserved2 = 0;
        
        if (!t->is_weight || !t->initializer) continue;
        
        Onnx__TensorProto *proto = t->initializer;
        
        // Get Data Pointer
        void *data_ptr = NULL;
        size_t data_len = 0;
        int need_free = 0;
        
        // Check raw_data first
        if (proto->raw_data.len > 0) {
            data_ptr = proto->raw_data.data;
            data_len = proto->raw_data.len;
        } else if (proto->n_float_data > 0) {
            data_len = proto->n_float_data * sizeof(float);
            data_ptr = malloc(data_len);
            if(data_ptr) {
                memcpy(data_ptr, proto->float_data, data_len);
                need_free = 1;
            }
        } else if (proto->n_int64_data > 0) {
            data_len = proto->n_int64_data * sizeof(int64_t);
            data_ptr = malloc(data_len);
            if(data_ptr) {
                memcpy(data_ptr, proto->int64_data, data_len);
                need_free = 1;
            }
        } else if (proto->n_int32_data > 0) {
            data_len = proto->n_int32_data * sizeof(int32_t);
            data_ptr = malloc(data_len);
            if(data_ptr) {
                memcpy(data_ptr, proto->int32_data, data_len);
                need_free = 1;
            }
        } else if (proto->n_external_data > 0) {
            data_ptr = read_external_data(proto, onnx_file, &data_len);
            if (data_ptr) need_free = 1;
        } else {
             fprintf(stderr, "Warning: Tensor %s payload not found (type=%d).\n", t->name, proto->data_type);
        }
        
        if (data_ptr && data_len > 0) {
            // Align start of each weight tensor to 4/8 bytes?
            // Let's just write packed. The WeightData block itself is aligned.
            
            weight_entries[i].data_offset = weight_relative_offset;
            weight_entries[i].data_size = data_len;
            
            fwrite(data_ptr, 1, data_len, fout);
            weight_relative_offset += data_len;
            
            if(need_free) free(data_ptr);
        }
    }
    
    // 4. Update Header & Offset Table
    header.total_weight_bytes = weight_relative_offset;
    header.reserved[0] = offset_table_pos; // Store Offset Table Offset in reserved[0]
    
    fseek(fout, 0, SEEK_SET);
    fwrite(&header, sizeof(header), 1, fout);
    
    fseek(fout, offset_table_pos, SEEK_SET);
    fwrite(weight_entries, sizeof(ENNF_WeightEntry), tensor_count, fout);
    
    // Done
    fclose(fout);
    free(weight_entries);
    free(buf);
    graph_opt_free(opt_ctx);
    onnx__model_proto__free_unpacked(model, NULL);
    
    printf("Successfully converted to %s\n", ennf_file);
    return 0;
}
