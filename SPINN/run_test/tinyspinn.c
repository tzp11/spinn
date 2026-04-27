/*
 * tinyspinn.c - ENNF 推理运行时 (适配 V2 格式)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "ennf_def.h"
#include "ennf_op_types.h"
#include "ennf_op_params.h"

// 模拟 Flash基地址
uint8_t *flash_base = NULL;

// 模拟 RAM (张量池)
typedef struct {
    void *data;
    uint32_t size;
    ENNF_TensorMeta meta;
} RuntimeTensor;

RuntimeTensor *tensor_pool = NULL;
uint32_t num_tensors = 0;

/* ============================================================
 * 简单算子实现 
 * ============================================================ */

void op_relu_impl(float *in, float *out, int count) {
    for (int i = 0; i < count; i++) {
        out[i] = (in[i] < 0) ? 0 : in[i];
    }
}

void op_reshape_impl(float *in, float *out, int count) {
    memcpy(out, in, count * sizeof(float));
}

void op_gemm_impl(float *A, float *B, float *C_bias, float *Out, 
             int M, int N) {
    // A: [1, M], B: [N, M] (transposed by default in ONNX Gemm usually if transB=1)
    // But wait, our helper extracts transB.
    // Assuming standard layout.
    for (int n = 0; n < N; n++) {
        float sum = 0;
        for (int m = 0; m < M; m++) {
            sum += A[m] * B[n * M + m];
        }
        Out[n] = sum + (C_bias ? C_bias[n] : 0);
    }
}

void op_conv_impl(float *x, float *w, float *y,
             int Cin, int Hin, int Win,
             int Cout, int K, int Stride) {
    
    int Hout = (Hin - K) / Stride + 1;
    int Wout = (Win - K) / Stride + 1;
    
    // printf("Conv: In[%d,%d,%d] K[%d] S[%d] -> Out[%d,%d,%d]\n", Cin, Hin, Win, K, Stride, Cout, Hout, Wout);
    
    memset(y, 0, Cout * Hout * Wout * sizeof(float));
    
    for (int c_out = 0; c_out < Cout; c_out++) {
        for (int h_out = 0; h_out < Hout; h_out++) {
            for (int w_out = 0; w_out < Wout; w_out++) {
                
                float sum = 0;
                int h_in_base = h_out * Stride;
                int w_in_base = w_out * Stride;
                
                for (int c_in = 0; c_in < Cin; c_in++) {
                    float *x_channel = x + c_in * Hin * Win;
                    float *w_kernel = w + (c_out * Cin + c_in) * K * K;
                    
                    for (int kh = 0; kh < K; kh++) {
                        for (int kw = 0; kw < K; kw++) {
                            sum += x_channel[(h_in_base + kh) * Win + (w_in_base + kw)] * 
                                   w_kernel[kh * K + kw];
                        }
                    }
                }
                y[c_out * Hout * Wout + h_out * Wout + w_out] = sum;
            }
        }
    }
}

/* ============================================================
 * Main
 * ============================================================ */

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <model.ennf>\n", argv[0]);
        return 1;
    }

    // 1. Load Model
    FILE *fp = fopen(argv[1], "rb");
    if (!fp) { perror("fopen"); return 1; }
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    flash_base = malloc(fsize);
    fread(flash_base, 1, fsize, fp);
    fclose(fp);

    ENNF_Header *header = (ENNF_Header*)flash_base;
    if (header->magic != ENNF_MAGIC) {
        printf("Invalid magic!\n");
        return 1;
    }
    printf("Model Loaded. Nodes: %d, Tensors: %d\n", header->num_nodes, header->num_tensors);

    // 2. Init Tensor Pool
    num_tensors = header->num_tensors;
    tensor_pool = calloc(num_tensors, sizeof(RuntimeTensor));

    // Load Tensor Meta
    uint8_t *ptr = flash_base + header->tensor_meta_offset;
    
    // We need offset table to locate weight data
    // Offset Table is now at header->reserved[0] (hack)
    // Actually, in our v2 writer, we wrote it there.
    uint32_t offset_table_pos = header->reserved[0];
    ENNF_WeightEntry *weight_table = (ENNF_WeightEntry*)(flash_base + offset_table_pos);
    
    // Iterate tensors
    for (int i = 0; i < num_tensors; i++) {
        ENNF_TensorMeta *meta = (ENNF_TensorMeta*)ptr;
        tensor_pool[meta->tensor_id].meta = *meta;
        
        if (meta->is_weight) {
            // Locate in Weight Data using O(1) Dense Table
            if (meta->tensor_id < header->num_tensors) {
                ENNF_WeightEntry *entry = &weight_table[meta->tensor_id];
                
                if (entry->data_size > 0) {
                    tensor_pool[meta->tensor_id].data = flash_base + header->weight_data_offset + entry->data_offset;
                    tensor_pool[meta->tensor_id].size = entry->data_size;
                    // printf("Tensor #%d (Weight): %p size %d\n", meta->tensor_id, tensor_pool[meta->tensor_id].data, entry->data_size);
                }
            }
        } 
        
        ptr += sizeof(ENNF_TensorMeta);
    }
    
    // 3. Prepare Input
    ptr = flash_base + header->graph_meta_offset;
    uint16_t n_in = *(uint16_t*)ptr; ptr += 2;
    uint16_t input_id = *(uint16_t*)ptr; // First input
    printf("Input Tensor ID: %d\n", input_id);
    
    RuntimeTensor *t_in = &tensor_pool[input_id];
    t_in->size = 1*3*32*32 * sizeof(float);
    t_in->data = malloc(t_in->size);
    // Manual Shape override for complex_test
    t_in->meta.dims[0] = 1;
    t_in->meta.dims[1] = 3;
    t_in->meta.dims[2] = 32;
    t_in->meta.dims[3] = 32;
    t_in->meta.elem_count = 1*3*32*32;
    
    float *indata = (float*)t_in->data;
    for(int k=0; k<t_in->meta.elem_count; k++) indata[k] = (float)k / 1000.0f; // tiny random-ish data

    // --- WEIGHT VERIFICATION BLOCK ---
    // Specifically check Tensor #0 (which is conv1.weight in complex_test)
    if (tensor_pool[0].data != NULL) {
        printf("\n[Weight Check] Tensor #0 Values in Memory:\n");
        float *wptr = (float*)tensor_pool[0].data;
        for(int k=0; k<8; k++) printf("%f ", wptr[k]);
        printf("\n\n");
    }
    // ---------------------------------
    
    // 4. Run Inference
    ptr = flash_base + header->node_table_offset;
    
    for (int i = 0; i < header->num_nodes; i++) {
        ENNF_NodeBase *node = (ENNF_NodeBase*)ptr;
        ptr += sizeof(ENNF_NodeBase);
        
        uint16_t *input_ids = (uint16_t*)ptr;
        ptr += node->num_inputs * 2;
        
        uint16_t *output_ids = (uint16_t*)ptr;
        ptr += node->num_outputs * 2;
        
        void *params = ptr;
        ptr += node->params_size; // Auto skip!
        
        // printf("Exec Node #%d Op: %d\n", i, node->op_type);

        /* Dispatch */
        if (node->op_type == OP_Conv) {
            ENNF_ConvParams *p = (ENNF_ConvParams*)params;
            ENNF_TensorMeta *xm = &tensor_pool[input_ids[0]].meta;
            ENNF_TensorMeta *wm = &tensor_pool[input_ids[1]].meta;
            ENNF_TensorMeta *ym = &tensor_pool[output_ids[0]].meta;
            
            int Hin = xm->dims[2];
            int Win = xm->dims[3];
            int K = wm->dims[2];
            int Stride = p->strides[0]; // assuming 2D symmetric
            if (Stride == 0) Stride = 1;
            int Cout = wm->dims[0];
            int Hout = (Hin - K) / Stride + 1;
            int Wout = (Win - K) / Stride + 1;
            
            // Set Output Meta
            ym->dims[0] = 1;
            ym->dims[1] = Cout;
            ym->dims[2] = Hout;
            ym->dims[3] = Wout;
            ym->elem_count = Cout * Hout * Wout;
            
            // Alloc
            if (!tensor_pool[output_ids[0]].data) {
                tensor_pool[output_ids[0]].size = ym->elem_count * 4;
                tensor_pool[output_ids[0]].data = calloc(1, tensor_pool[output_ids[0]].size);
            }
            
            op_conv_impl(tensor_pool[input_ids[0]].data, 
                         tensor_pool[input_ids[1]].data, 
                         tensor_pool[output_ids[0]].data,
                         wm->dims[1], Hin, Win, Cout, K, Stride);
                         
        } else if (node->op_type == OP_Relu) {
            ENNF_TensorMeta *xm = &tensor_pool[input_ids[0]].meta;
            ENNF_TensorMeta *ym = &tensor_pool[output_ids[0]].meta;
            *ym = *xm; ym->tensor_id = output_ids[0];
            
            if (!tensor_pool[output_ids[0]].data) {
                tensor_pool[output_ids[0]].size = ym->elem_count * 4;
                tensor_pool[output_ids[0]].data = calloc(1, tensor_pool[output_ids[0]].size);
            }
            op_relu_impl(tensor_pool[input_ids[0]].data, 
                         tensor_pool[output_ids[0]].data, xm->elem_count);
                         
        } else if (node->op_type == OP_Reshape) {
            // params_size is 0. Shape is from input_ids[1]
            ENNF_TensorMeta *xm = &tensor_pool[input_ids[0]].meta;
            ENNF_TensorMeta *ym = &tensor_pool[output_ids[0]].meta;
            ym->elem_count = xm->elem_count; // Just trust total count
            
            if (!tensor_pool[output_ids[0]].data) {
                tensor_pool[output_ids[0]].size = xm->elem_count * 4;
                tensor_pool[output_ids[0]].data = calloc(1, tensor_pool[output_ids[0]].size);
            }
            op_reshape_impl(tensor_pool[input_ids[0]].data, 
                            tensor_pool[output_ids[0]].data, xm->elem_count);
                            
        } else if (node->op_type == OP_Gemm) {
            ENNF_GemmParams *p = (ENNF_GemmParams*)params;
            ENNF_TensorMeta *wm = &tensor_pool[input_ids[1]].meta;
            ENNF_TensorMeta *ym = &tensor_pool[output_ids[0]].meta;
            
            int M = wm->dims[1]; 
            int N = wm->dims[0]; 
            ym->dims[0] = 1;
            ym->dims[1] = N;
            ym->elem_count = N;
            
             if (!tensor_pool[output_ids[0]].data) {
                tensor_pool[output_ids[0]].size = N * 4;
                tensor_pool[output_ids[0]].data = calloc(1, tensor_pool[output_ids[0]].size);
            }
            
            float *bias = (node->num_inputs > 2) ? tensor_pool[input_ids[2]].data : NULL;
            op_gemm_impl(tensor_pool[input_ids[0]].data, 
                         tensor_pool[input_ids[1]].data, 
                         bias,
                         tensor_pool[output_ids[0]].data, M, N);
        } else {
             printf("  -> Op %d not implemented yet (Skipping execution). ParamsSize=%d\n", node->op_type, node->params_size);
             if (node->params_size > 0 && node->params_size < 64) {
                 printf("     Params Hex: ");
                 uint8_t *p8 = (uint8_t*)params;
                 for(int k=0; k<node->params_size; k++) printf("%02x ", p8[k]);
                 printf("\n");
             }
             
             // CRITICAL FIX: Allocate memory for outputs even if skipped, 
             // so downstream nodes don't SEGV on NULL pointers.
             for (int j=0; j<node->num_outputs; j++) {
                 uint16_t out_id = output_ids[j];
                 if (tensor_pool[out_id].data == NULL) {
                     size_t sz = tensor_pool[out_id].size;
                     if (sz == 0) sz = 1024; // Fallback for dynamic/bad shapes
                     tensor_pool[out_id].data = calloc(1, sz);
                     // printf("    -> Allocated dummy output #%d (%zu bytes)\n", out_id, sz);
                 }
             }
        }
    }
    
    // 5. Print Result
    // Assume last node output is result
    // Or we look at graph outputs
    ptr = flash_base + header->graph_meta_offset;
    ptr += 2 + n_in*2; // skip inputs
    uint16_t n_out = *(uint16_t*)ptr; ptr += 2;
    uint16_t out_id = *(uint16_t*)ptr;
    
    float *res = (float*)tensor_pool[out_id].data;
    printf("Result:\n");
    for(int k=0; k<10; k++) printf("%f ", res[k]);
    printf("\n");

    return 0;
}
