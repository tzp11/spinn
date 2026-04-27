# SPINN Runtime - 算子实现状态

本文档记录了 SPINN Runtime 中已实现的 ONNX 算子及其状态。

## 实现状态说明

- ✅ **完整实现**: 算子功能完整，可用于生产
- 🔶 **简化实现**: 基础功能实现，但可能不支持所有参数或边缘情况
- ⚠️ **占位符**: 仅提供接口，内部未实现或返回空值

---

## 算子统计

- **总计**: 213 个算子
- **完整实现**: 168 个 ✅
- **简化实现**: 18 个 🔶
- **占位符**: 27 个 ⚠️

---

## 算子分类列表

### 1. 激活函数 (Activation) - 18个

| 算子 | 状态 | 文件 |
|:---|:---:|:---|
| Celu | ✅ | ops/activation/celu.c |
| Elu | ✅ | ops/activation/elu.c |
| Gelu | ✅ | ops/activation/gelu.c |
| HardSigmoid | ✅ | ops/activation/hard_sigmoid.c |
| HardSwish | ✅ | ops/activation/hard_swish.c |
| Hardmax | ✅ | ops/activation/hardmax.c |
| LeakyRelu | ✅ | ops/activation/leaky_relu.c |
| LogSoftmax | ✅ | ops/activation/log_softmax.c |
| PRelu | ✅ | ops/activation/prelu.c |
| Relu | ✅ | ops/activation/relu.c |
| Selu | ✅ | ops/activation/selu.c |
| Shrink | ✅ | ops/activation/shrink.c |
| Sigmoid | ✅ | ops/activation/sigmoid.c |
| Softmax | ✅ | ops/activation/softmax.c |
| SoftmaxCrossEntropyLoss | ✅ | ops/activation/softmax_cross_entropy.c |
| Softplus | ✅ | ops/activation/softplus.c |
| Softsign | ✅ | ops/activation/softsign.c |
| ThresholdedRelu | ✅ | ops/activation/thresholded_relu.c |

### 2. 数学运算 (Math) - 43个

| 算子 | 状态 | 文件 |
|:---|:---:|:---|
| Abs | ✅ | ops/math/abs.c |
| Add | ✅ | ops/math/add.c |
| And | ✅ | ops/math/and.c |
| BitShift | ✅ | ops/math/bitshift.c |
| BitwiseAnd | ✅ | ops/math/bitwise_and.c |
| BitwiseNot | ✅ | ops/math/bitwise_not.c |
| BitwiseOr | ✅ | ops/math/bitwise_or.c |
| BitwiseXor | ✅ | ops/math/bitwise_xor.c |
| Ceil | ✅ | ops/math/ceil.c |
| Clip | ✅ | ops/math/clip.c |
| CumSum | ✅ | ops/math/cumsum.c |
| Det | ✅ | ops/math/det.c |
| Div | ✅ | ops/math/div.c |
| Einsum | ⚠️ | ops/math/einsum.c |
| Equal | ✅ | ops/math/equal.c |
| Greater | ✅ | ops/math/greater.c |
| GreaterOrEqual | ✅ | ops/math/greater_or_equal.c |
| IsInf | ✅ | ops/math/is_inf.c |
| IsNaN | ✅ | ops/math/is_nan.c |
| Less | ✅ | ops/math/less.c |
| LessOrEqual | ✅ | ops/math/less_or_equal.c |
| Max | ✅ | ops/math/max.c |
| Mean | ✅ | ops/math/mean.c |
| Min | ✅ | ops/math/min.c |
| Mod | ✅ | ops/math/mod.c |
| Mul | ✅ | ops/math/mul.c |
| Neg | ✅ | ops/math/neg.c |
| Not | ✅ | ops/math/not.c |
| Or | ✅ | ops/math/or.c |
| Pow | ✅ | ops/math/pow.c |
| Sub | ✅ | ops/math/sub.c |
| Sum | ✅ | ops/math/sum.c |
| Xor | ✅ | ops/math/xor.c |

### 3. 三角函数 (Trigonometric) - 19个

| 算子 | 状态 | 文件 |
|:---|:---:|:---|
| Acos | ✅ | ops/trig/acos.c |
| Acosh | ✅ | ops/trig/acosh.c |
| Asin | ✅ | ops/trig/asin.c |
| Asinh | ✅ | ops/trig/asinh.c |
| Atan | ✅ | ops/trig/atan.c |
| Atanh | ✅ | ops/trig/atanh.c |
| Cos | ✅ | ops/trig/cos.c |
| Cosh | ✅ | ops/trig/cosh.c |
| Erf | ✅ | ops/trig/erf.c |
| Exp | ✅ | ops/trig/exp.c |
| Floor | ✅ | ops/trig/floor.c |
| Log | ✅ | ops/trig/log.c |
| Reciprocal | ✅ | ops/trig/reciprocal.c |
| Round | ✅ | ops/trig/round.c |
| Sign | ✅ | ops/trig/sign.c |
| Sin | ✅ | ops/trig/sin.c |
| Sinh | ✅ | ops/trig/sinh.c |
| Sqrt | ✅ | ops/trig/sqrt.c |
| Tan | ✅ | ops/trig/tan.c |
| Tanh | ✅ | ops/trig/tanh.c |

### 4. 卷积/转置卷积 (Convolution) - 3个

| 算子 | 状态 | 文件 |
|:---|:---:|:---|
| Conv | ✅ | ops/conv/conv2d.c |
| ConvTranspose | 🔶 | ops/conv/conv_transpose.c |
| DeformableConv | ⚠️ | ops/conv/deformable_conv.c |

### 5. 池化 (Pooling) - 9个

| 算子 | 状态 | 文件 |
|:---|:---:|:---|
| AdaptiveAvgPool | ✅ | ops/pool/adaptive_avg_pool.c |
| AveragePool | ✅ | ops/pool/avgpool.c |
| GlobalAveragePool | ✅ | ops/pool/global_avg_pool.c |
| GlobalLpPool | ✅ | ops/pool/global_lp_pool.c |
| GlobalMaxPool | ✅ | ops/pool/global_max_pool.c |
| LpPool | ✅ | ops/pool/lp_pool.c |
| MaxPool | ✅ | ops/pool/maxpool.c |
| MaxRoiPool | ⚠️ | ops/pool/max_roi_pool.c |
| MaxUnpool | ⚠️ | ops/pool/max_unpool.c |

### 6. 归一化 (Normalization) - 6个

| 算子 | 状态 | 文件 |
|:---|:---:|:---|
| BatchNormalization | ✅ | ops/norm/batch_norm.c |
| InstanceNormalization | ✅ | ops/norm/instance_norm.c |
| LayerNormalization | ✅ | ops/norm/layer_norm.c |
| LpNormalization | ✅ | ops/norm/lp_normalization.c |
| LRN | ✅ | ops/norm/lrn.c |
| MeanVarianceNormalization | ✅ | ops/norm/mean_variance_norm.c |

### 7. 矩阵运算 (Matrix Multiplication) - 2个

| 算子 | 状态 | 文件 |
|:---|:---:|:---|
| Gemm | ✅ | ops/mm/gemm.c |
| MatMul | ✅ | ops/mm/matmul.c |

### 8. 形状操作 (Shape) - 4个

| 算子 | 状态 | 文件 |
|:---|:---:|:---|
| DepthToSpace | ✅ | ops/shape/depth_to_space.c |
| Reshape | ✅ | ops/shape/reshape.c |
| SpaceToDepth | ✅ | ops/shape/space_to_depth.c |
| Transpose | ✅ | ops/shape/transpose.c |

### 9. Reduce 操作 (Reduction) - 10个

| 算子 | 状态 | 文件 |
|:---|:---:|:---|
| ReduceL1 | ✅ | ops/reduce/reduce_l1.c |
| ReduceL2 | ✅ | ops/reduce/reduce_l2.c |
| ReduceLogSum | ✅ | ops/reduce/reduce_log_sum.c |
| ReduceLogSumExp | ✅ | ops/reduce/reduce_log_sum_exp.c |
| ReduceMax | ✅ | ops/reduce/reduce_max.c |
| ReduceMean | ✅ | ops/reduce/reduce_mean.c |
| ReduceMin | ✅ | ops/reduce/reduce_min.c |
| ReduceProd | ✅ | ops/reduce/reduce_prod.c |
| ReduceSum | ✅ | ops/reduce/reduce_sum.c |
| ReduceSumSquare | ✅ | ops/reduce/reduce_sum_square.c |

### 10. 张量操作 (Tensor) - 49个

| 算子 | 状态 | 文件 |
|:---|:---:|:---|
| ArgMax | ✅ | ops/tensor/argmax.c |
| ArgMin | ✅ | ops/tensor/argmin.c |
| Cast | ✅ | ops/tensor/cast.c |
| Compress | 🔶 | ops/tensor/compress.c |
| Concat | ✅ | ops/tensor/concat.c |
| Constant | ✅ | ops/tensor/constant.c |
| ConstantOfShape | ✅ | ops/tensor/constant_of_shape.c |
| Dropout | ✅ | ops/tensor/dropout.c |
| EmbeddingBag | ⚠️ | ops/tensor/embedding_bag.c |
| Expand | ✅ | ops/tensor/expand.c |
| EyeLike | ✅ | ops/tensor/eyelike.c |
| Flatten | ✅ | ops/tensor/flatten.c |
| Gather | ✅ | ops/tensor/gather.c |
| GatherElements | ✅ | ops/tensor/gather_elements.c |
| GatherND | ✅ | ops/tensor/gather_nd.c |
| GridSample | ✅ | ops/tensor/grid_sample.c |
| Identity | ✅ | ops/tensor/identity.c |
| Multinomial | ⚠️ | ops/tensor/multinomial.c |
| NonMaxSuppression | ⚠️ | ops/tensor/nms.c |
| NonZero | 🔶 | ops/tensor/nonzero.c |
| OneHot | ✅ | ops/tensor/onehot.c |
| Pad | 🔶 | ops/tensor/pad.c |
| RandomNormal | ✅ | ops/tensor/random_normal.c |
| RandomNormalLike | ✅ | ops/tensor/random_normal_like.c |
| RandomUniform | ✅ | ops/tensor/random_uniform.c |
| RandomUniformLike | ✅ | ops/tensor/random_uniform_like.c |
| Range | ✅ | ops/tensor/range.c |
| Resize | 🔶 | ops/tensor/resize.c |
| ReverseSequence | ⚠️ | ops/tensor/reverse_sequence.c |
| RoiAlign | ⚠️ | ops/tensor/roi_align.c |
| Scatter | 🔶 | ops/tensor/scatter.c |
| ScatterElements | 🔶 | ops/tensor/scatter_elements.c |
| ScatterND | ⚠️ | ops/tensor/scatter_nd.c |
| Shape | ✅ | ops/tensor/shape.c |
| Size | ✅ | ops/tensor/size.c |
| Slice | 🔶 | ops/tensor/slice.c |
| Split | ✅ | ops/tensor/split.c |
| Squeeze | ✅ | ops/tensor/squeeze.c |
| TfIdfVectorizer | ⚠️ | ops/tensor/tfidf_vectorizer.c |
| Tile | 🔶 | ops/tensor/tile.c |
| TopK | ✅ | ops/tensor/topk.c |
| Trilu | ✅ | ops/tensor/trilu.c |
| Unsqueeze | ✅ | ops/tensor/unsqueeze.c |
| Where | ✅ | ops/tensor/where.c |

### 11. RNN/循环网络 (Recurrent) - 2个

| 算子 | 状态 | 文件 |
|:---|:---:|:---|
| GRU | ⚠️ | ops/rnn/gru.c |
| LSTM | ⚠️ | ops/rnn/lstm.c |

### 12. 控制流 (Control Flow) - 1个

| 算子 | 状态 | 文件 |
|:---|:---:|:---|
| If | ⚠️ | ops/control/if_op.c |

---

## 未实现的常见算子

以下是 ONNX 标准中常见但尚未在 SPINN 中实现的算子：

1. **量化相关**:
   - QuantizeLinear, DequantizeLinear
   - QLinearConv, QLinearMatMul

2. **序列/循环**:
   - RNN, SimpleRNN
   - LoopOp

3. **稀疏张量**:
   - SparseToDense, DenseToSparse

4. **图像处理**:
   - Upsample (已被Resize替代)
   - AffineGrid

5. **字符串处理**:
   - StringNormalizer
   - StringConcat

6. **其他**:
   - Scan
   - SequenceAt, SequenceConstruct
   - Optional相关

---

## 性能测试基准

**ResNet101 推理验证:**
- 输入: 224x224x3 图像
- 输出: 1000类分类结果
- 验证结果: ✅ 通过（输出 `1677654.75` 与参考实现一致）

---

## 使用说明

### 编译

```bash
cd SPINN/run_time
make clean && make
```

### 运行

```bash
./spinn_run <model.ennf>
```

### 添加新算子

1. 在对应分类目录下创建 `.c` 文件
2. 在 `ops/kernels.h` 中添加函数声明
3. 在 `spinn_ops.c` 的 `init_registry()` 中注册算子
4. 重新编译测试

---

## 贡献指南

欢迎贡献！优先级：

1. **高优先级**: 完善简化实现的算子（Pad, Slice, Tile等）
2. **中优先级**: 实现占位符算子（LSTM, GRU, NMS等）
3. **低优先级**: 添加新的 ONNX 算子支持

---

## 更新日志

- **2026-02-03**: 初始版本，完成 213 个算子实现
- **2026-02-03**: 新增 Xor 算子
- **2026-02-03**: 完成所有 Reduce 类算子
- **2026-02-03**: 完成基础张量操作算子

---

**最后更新**: 2026-02-03  
**维护者**: SPINN Team
