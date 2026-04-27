/*
 * kernels.h - SPINN Operator Kernels Declarations
 * Updated with new operators from image
 */

#ifndef __SPINN_KERNELS_H__
#define __SPINN_KERNELS_H__

#include "../spinn_ops.h"
#include <stdint.h>

// ============================================================
// Activation Functions (ops/activation/)
// ============================================================
int op_relu(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_softmax(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_sigmoid(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_celu(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_elu(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_leaky_relu(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_selu(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_prelu(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_hard_sigmoid(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_hard_swish(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_gelu(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);

// ============================================================
// Math / Elementwise Operations (ops/math/)
// ============================================================
int op_abs(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_add(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_and(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_bitwise_and(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_bitwise_not(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_bitwise_or(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_bitwise_xor(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_ceil(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_clip(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_cumsum(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_div(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_det(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_equal(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_mul(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_neg(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_pow(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_sub(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);

// ============================================================
// Trigonometric / Math Functions (ops/trig/)
// ============================================================
int op_acos(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_acosh(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_asin(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_asinh(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_atan(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_atanh(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_cos(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_cosh(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_erf(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_exp(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_floor(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_log(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_reciprocal(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_round(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_sign(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_sin(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_sinh(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_sqrt(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_tan(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_tanh(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);

// ============================================================
// Matrix Operations (ops/mm/)
// ============================================================
int op_matmul(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_gemm(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);

// ============================================================
// Convolution (ops/conv/)
// ============================================================
int op_conv2d(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_conv_transpose(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);

// ============================================================
// Pooling (ops/pool/)
// ============================================================
int op_maxpool(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_avgpool(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_global_avg_pool(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_global_max_pool(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);

// ============================================================
// Normalization (ops/norm/)
// ============================================================
int op_batch_norm(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_instance_norm(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_layer_norm(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);

// ============================================================
// Shape / Transform (ops/shape/)
// ============================================================
int op_depth_to_space(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_reshape(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_transpose(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_constant_of_shape(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);

// ============================================================
// Reduction (ops/reduce/)
// ============================================================
int op_reduce_mean(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);

// ============================================================
// Tensor Operations (ops/tensor/)
// ============================================================
int op_argmax(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_argmin(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_cast(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_concat(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_constant(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_dropout(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_expand(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_eyelike(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_flatten(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_gather(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_gather_elements(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_gather_nd(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_identity(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_split(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_squeeze(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_unsqueeze(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_bitshift(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_adaptive_avg_pool(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_compress(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_concat(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_constant(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_constant_of_shape(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_cumsum(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_conv_transpose(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_deformable_conv(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_depth_to_space(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_det(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_dropout(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_einsum(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_embedding_bag(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_expand(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_eyelike(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_gather_elements(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_global_lp_pool(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_greater(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_greater_or_equal(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_grid_sample(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_gru(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_hardmax(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_if(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_is_inf(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_is_nan(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_less(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_less_or_equal(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_log_softmax(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_lp_normalization(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_lp_pool(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_lrn(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_lstm(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_max(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_max_roi_pool(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_max_unpool(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_mean(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_mean_variance_norm(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_min(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_mod(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_multinomial(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_non_max_suppression(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_nonzero(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_not(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_onehot(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_or(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_pad(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_random_normal(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_random_normal_like(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_random_uniform(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_random_uniform_like(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_range(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_reduce_l1(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_reduce_l2(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_reduce_log_sum(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_reduce_log_sum_exp(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_reduce_max(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_reduce_min(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_reduce_prod(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_reduce_sum(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_reduce_sum_square(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_resize(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_reverse_sequence(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_roi_align(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_scatter(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_scatter_elements(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_scatter_nd(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_shape(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_shrink(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_size(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_slice(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_softmax_cross_entropy_loss(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_softplus(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_softsign(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_space_to_depth(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_sum(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_tfidf_vectorizer(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_thresholded_relu(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_tile(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_topk(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_trilu(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_where(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);
int op_xor(SpinnTensor **in, int n_in, void *params, uint16_t params_size, SpinnTensor **out, int n_out);

#endif
