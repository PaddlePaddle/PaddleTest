# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
parallel execute cases in paddle ci
"""
import re
import time
import queue
import threading
import os
import json
import sys

taskQueue = queue.Queue()
lock = threading.RLock()

failed_ce_case_list = []
ignore_case_dir = {
    "device": [
        "test_cuda_empty_cache.py", # passed
        "test_cuda_get_device_name.py", # passed
        "test_cuda_memory.py", # passed
        "test_device_count.py", # passed
        "test_cuda_get_device_capability.py", # passed
        "test_cuda_get_device_properites.py", # passed
        "test_cuda_stream_guard.py", # passed
    ],
    "fft": [],
    "incubate": [
        "test_jvp.py", # passed
        "test_Jacobian.py",
        "test_vjp.py",
        "test_fused_feedforward.py",
        "test_FusedFeedForward.py",
        "test_FusedTransformerEncoderLayer.py",
        "test_graph_send_recv.py", # passed
        "test_Hessian.py",
        "test_incubate_softmax_mask_fuse_upper_triangle.py", # passed
        "test_incubate_softmax_mask_fuse.py",
        "test_minimize_bfgs.py",
        "test_minimize_lbfgs.py",
        "test_prim_status.py", # passed
        "test_prim2orig.py",
        "test_sparse_coo_tensor.py", # passed
        "test_sparse_csr_tensor.py", # passed
        "test_FusedMultiHeadAttention.py",
    ],
    "linalg": [
        "test_det.py", # passed
        "test_cov.py", # passed
        "test_eigvals.py", # passed
        "test_norm.py",
        "test_eig.py", # passed
        "test_eigh.py",
        "test_lu_unpack.py",
        "test_lu.py",
        "test_cholesky_solve.py",
        "test_cholesky.py",
        "test_inv.py",
        "test_eigvalsh.py",
        "test_cond.py",
        "test_multi_dot.py",
        "test_matrix_rank.py",
        "test_matrix_power.py",
        "test_slogdet.py",
        "test_qr.py",
        "test_pinv.py",
        "test_svd.py",
        "test_triangular_solve.py",
        "test_solve.py",
    ],
    "loss": [
        "test_margin_cross_entropy.py", # passed
        # "test_softmax_with_cross_entropy.py", # passed
        "test_kl_div.py", # passed
        "test_nn_cross_entropy_loss_3.py", # passed
        "test_log_loss.py", # passed
        "test_npair_loss.py", # passed
        "test_sigmoid_focal_loss.py", # passed
        "test_binary_cross_entropy_with_logits.py",
        "test_cross_entropy.py",
        "test_l1_loss.py",
        "test_dice_loss.py",
        "test_margin_cross_entropy_1.py", # passed
        "test_ctc_loss.py",
        "test_margin_cross_entropy_2.py", # passed
        "test_ctc_loss_1.py", # passed
        "test_binary_cross_entropy.py",
        "test_margin_ranking_loss.py",
        "test_HingeEmbeddingLoss.py", # passed
        "test_margin_cross_entropy_3.py", # passed
        "test_mse_loss.py",
        "test_nn_cross_entropy_loss_1.py",
        "test_nn_bce_loss.py",
        "test_nn_ctc_loss.py",
        "test_nn_bce_with_logits_loss.py",
        "test_nn_cross_entropy_loss_2.py",
        "test_nll_loss.py",
        "test_nn_ctc_loss_1.py", # passed
        "test_nn_cross_entropy_loss.py",
        "test_nn_kldiv_loss.py", # passed
        "test_nn_l1_loss.py",
        "test_nn_margin_ranking_loss.py",
        "test_nn_mse_loss.py",
        "test_nn_nll_loss.py",
        "test_nn_smooth_l1_loss.py", # passed
        "test_square_error_cost.py", # passed
        "test_smooth_l1_loss.py", # passed
    ],
    "nn": [
        "test_alpha_dropout.py", # passed
        "test_functional_conv3d.py",
        "test_functional_softshrink.py",
        "test_layernorm.py", # passed
        "test_avg_pool1D.py",
        "test_functional_conv3d_float64.py",
        "test_functional_softsign.py", # passed
        "test_layernorm_float64.py",
        "test_avg_pool2D.py",
        "test_functional_cosine_similarity.py", # passed
        "test_functional_swish.py", # passed
        "test_leakyrelu.py", # passed
        "test_avg_pool3D.py", # passed
        "test_functional_dropout.py", # passed
        "test_functional_tanhshrink.py", # passed
        "test_logsigmoid.py", # passed
        "test_batchnorm.py", # passed
        "test_functional_dropout2d.py", # passed
        "test_functional_temporal_shift.py", # passed
        "test_logsoftmax.py",
        "test_batchnorm1D.py", # passed
        "test_functional_dropout3d.py", # passed
        "test_functional_thresholded_relu.py", # passed
        "test_max_pool1d.py", # passed
        "test_functional_elu.py",
        "test_functional_transpose_conv1d.py",
        "test_max_pool2d.py",
        "rnn_numpy.py", # passed
        "test_batchnorm3D.py",
        "test_functional_elu_.py", # passed
        "test_functional_transpose_conv1d_float64.py",
        "test_max_pool3d.py", # passed
        "test_beamsearchdecoder.py", # passed
        "test_functional_embedding.py",
        "test_functional_transpose_conv2d.py",
        "test_maxout.py",
        "test_Bilinear.py", # passed
        "test_birnn.py", # passed
        "test_functional_fold.py", # passed
        "test_functional_transpose_conv2d_float64.py",
        "test_maxunpool2d.py", # passed
        "test_CELU.py",
        "test_clip_grad_by_global_norm.py", # passed
        "test_functional_gelu.py", # passed
        "test_functional_transpose_conv3d.py",
        "test_pixel_shuffle.py", # passed
        "test_Dropout.py", # passed
        "test_clip_grad_by_norm.py",
        "test_functional_glu.py", # passed
        "test_functional_transpose_conv3d_float64.py",
        "test_prelu.py", # passed
        "test_Dropout2D.py", # passed
        "test_clip_grad_by_value.py", # passed
        "test_functional_gumbel_softmax.py",
        "test_functional_unfold.py", # passed
        "test_relu.py", # passed
        "test_Dropout3D.py", # passed
        "test_conv1d.py", # passed
        "test_functional_hardshrink.py",
        "test_functional_upsample.py", # passed
        "test_relu6.py", # passed
        "test_Embedding.py", # passed
        "test_conv1d_float64.py",
        "test_functional_hardswish.py", # passed
        "test_functional_utils_remove_weight_norm.py", # passed
        "test_rnn.py", # passed
        "test_GRU.py", # passed
        "test_conv2d.py", # passed
        "test_functional_hardtanh.py",
        "test_functional_utils_weight_norm.py", # passed
        "test_rnncellbase.py",
        "test_GRUCell.py", # passed
        "test_conv2d_float64.py",
        "test_functional_instance_norm.py", # passed
        "test_gelu.py", # passed
        "test_selu.py", # passed
        "test_HingeEmbeddingLoss.py", # passed
        "test_conv3d.py",
        "test_functional_interpolate.py", # passed
        "test_group_norm.py", # passed
        # "test_silu.py", # passed
        "test_Identity.py", # passed
        "test_conv3d_float64.py",
        "test_functional_layer_norm.py", # passed
        "test_hardshrink.py",
        "test_softmax.py", # passed
        "test_LSTM.py", # passed
        "test_cosinesimilarity.py",
        "test_functional_layer_norm_float64.py",
        "test_hardsigmoid.py", # passed
        "test_softplus.py",
        "test_LSTMCell.py", # passed
        "test_dynamicdecode.py", # passed
        "test_functional_leaky_relu.py",
        "test_hardswish.py", # passed
        "test_softshrink.py", # passed
        "test_Linear.py", # passed
        "test_elu.py", # passed
        "test_functional_logsigmoid.py", # passed
        "test_hardtanh.py",
        "test_softsign.py", # passed
        "test_MultiHeadAttention.py", # passed
        "test_flatten.py", # passed
        "test_functional_logsoftmax.py",
        "test_initializer_Dirac.py", # passed
        "test_spectral_norm.py", # passed
        "test_Pad1D.py", # passed
        "test_functional_adaptive_avg_pool1d.py",
        "test_functional_max_pool1d.py", # passed
        "test_initializer_Orthogonal.py",
        "test_swish.py", # passed
        "test_Pad2D.py", # passed
        "test_functional_adaptive_avg_pool2d.py", # passed
        "test_functional_max_pool2d.py",
        "test_initializer_assign.py", # passed
        "test_tanh.py", # passed
        "test_Pad3D.py", # passed
        "test_functional_adaptive_avg_pool3d.py",
        "test_functional_max_pool3d.py", # passed
        "test_initializer_bilinear.py",
        "test_tanhshrink.py", # passed
        "test_PairwiseDistance.py", # passed
        "test_functional_adaptive_max_pool1d.py", # passed
        "test_functional_maxout.py",
        "test_initializer_calculate_gain.py", # passed
        "test_thresholdedrelu.py", # passed
        "test_Sigmoid.py", # passed
        "test_functional_adaptive_max_pool2d.py", # passed
        "test_functional_normalize.py", # passed
        "test_initializer_kaiming_normal.py", # passed
        "test_transpose_conv1d.py",
        "test_SimpleRNN.py",
        "test_functional_adaptive_max_pool3d.py", # passed
        "test_functional_one_hot.py", # passed
        "test_initializer_kaiming_uniform.py", # passed
        "test_transpose_conv1d_float64.py",
        "test_SimpleRNNCell.py", # passed
        "test_functional_affine_grid.py",
        "test_functional_pad.py", # passed
        "test_initializer_normal.py", # passed
        "test_transpose_conv2d.py",
        "test_SyncBatchNorm.py", # passed
        "test_functional_avg_pool1d.py",
        "test_functional_pixel_shuffle.py", # passed
        "test_initializer_normal_new.py", # passed
        "test_transpose_conv2d_float64.py",
        "test_Transformer.py", # passed
        "test_functional_avg_pool2d.py",
        "test_functional_prelu.py", # passed
        "test_initializer_truncated_normal.py",
        "test_transpose_conv3d.py",
        "test_TransformerDecoder.py", # passed
        "test_functional_avg_pool3d.py", # passed
        "test_functional_relu.py",
        "test_initializer_truncated_normal_new.py",
        "test_transpose_conv3d_float64.py",
        "test_TransformerDecoderLayer.py", # passed
        "test_functional_batch_norm.py", # passed
        "test_functional_relu6.py", # passed
        "test_initializer_uniform_new.py", # passed
        "test_unfold.py", # passed
        "test_TransformerEncoder.py", # passed
        "test_functional_celu.py",
        "test_functional_relu_.py", # passed
        "test_initializer_xavier_normal_new.py", # passed
        "test_upsample.py", # hang
        "test_TransformerEncoderLayer.py", # passed
        "test_functional_class_center_sample.py", # passed
        "test_functional_selu.py", # passed
        "test_initializer_xavier_uniform_new.py", # passed
        "test_upsampling_bilinear_2d.py", # hang
        "test_adaptive_avg_pool1D.py",
        "test_functional_conv1d.py", # passed
        "test_functional_sequence_mask.py", # passed
        "test_instance_norm1D.py",
        "test_upsamplingnearest2d.py", # hang
        "test_adaptive_avg_pool2D.py", # passed
        "test_functional_conv1d_float64.py",
        "test_functional_sigmoid.py", # passed
        "test_instance_norm2D.py", # passed
        "test_utils_parameters_to_vector.py", # passed
        "test_adaptive_avg_pool3D.py", # long time
        "test_functional_conv1d_transpose.py",
        "test_functional_silu.py", # passed
        "test_instance_norm2D_float64.py",
        "test_utils_spectral_norm.py", # passed
        "test_adaptive_max_pool1D.py", # passed
        "test_functional_conv2d.py", # passed
        "test_functional_softmax.py", # passed
        "test_instance_norm3D.py", # passed
        "test_utils_vector_to_parameters.py", # passed
        "test_adaptive_max_pool2D.py", # passed
        "test_functional_conv2d_float64.py",
        "test_functional_softmax_.py", # passed
        "test_instance_norm3D_float64.py",
        "test_adaptive_max_pool3D.py", # passed
        "test_functional_conv2d_transpose.py",
        "test_functional_softplus.py",
        "test_layer_norm.py",  # passed
        "test_batchnorm2D.py", # passed
    ],
    "paddlebase": [
        "test_acos.py", # passed
        "test_ceil.py", # passed
        "test_expand_as.py",
        "test_isfinite.py",
        "test_moveaxis.py", # passed
        "test_reshape.py", # passed
        "test_tanh.py", # passed
        # "test_add.py", # passed
        "test_chunk.py", # passed
        "test_expm1.py",
        "test_isinf.py",
        "test_multinomial.py", # passed
        "test_reshape_.py", # passed
        "test_tanh_.py", # passed
        "test_add_n.py", # passed
        "test_clip.py", # passed
        "test_eye.py",
        "test_isnan.py",
        "test_multiplex.py", # passed
        "test_roll.py", # passed
        "test_tensordot.py",
        "test_addmm.py",
        "test_clone.py", # passed
        "test_fill_.py", # passed
        "test_kron.py", # passed
        "test_multiply.py", # passed
        "test_rot90.py", # passed
        "test_tile.py",
        "test_all.py", # passed
        "test_complex.py", # passed
        "test_fill_diagonal_.py", # passed
        "test_kthvalue.py", # passed
        "test_mv.py", # passed
        "test_round.py", # passed
        "test_to_tensor.py", # passed
        "test_allclose.py",
        "test_concat.py", # passed
        "test_fill_diagonal_tensor.py", # passed
        "test_lerp.py", # passed
        "test_nansum.py", # passed
        # "test_rsqrt.py", # passed
        "test_tolist.py", # passed
        "test_T.py", # passed
        "test_amax.py", # passed
        "test_conj.py",
        "test_flatten.py", # passed
        "test_less_equal.py", # passed
        "test_neg.py", # passed
        "test_save_inference_model.py", # passed
        "test_topk.py",
        "test_Tensor_amax.py", # passed
        "test_amin.py", # passed
        # "test_cos.py", # passed
        "test_flip.py", # passed
        "test_less_than.py", # passed
        "test_nonzero.py", # passed
        "test_scale.py", # passed
        "test_trace.py", # passed
        "test_Tensor_amin.py", # passed
        # "test_any.py", # passed
        "test_cosh.py", # passed
        "test_floor.py", # passed
        "test_lgamma.py",
        "test_normal.py", # passed
        "test_scatter_nd_add.py",
        "test_transpose.py", # passed
        "test_Tensor_cholesky_solve.py",
        "test_arange.py", # passed
        "test_crop.py", # passed
        "test_floor_divide.py", # passed
        "test_linspace.py", # passed
        # "test_not_equal.py", # passed
        "test_searchsorted.py", # passed
        # "test_tril.py", # passed
        "test_Tensor_diff.py", # passed
        "test_argmax.py",
        "test_cross.py",
        "test_full.py", # passed
        "test_log.py",
        "test_numel.py", # passed
        "test_shape.py", # passed
        "test_triu.py", # passed
        "test_Tensor_element_size.py",
        "test_argmin.py",
        "test_cumprod.py",
        "test_full_like.py", # passed
        "test_log10.py", # passed
        "test_ones.py",
        "test_shard_index.py", # passed
        "test_trunc.py", # passed
        "test_Tensor_erfinv.py",
        "test_argsort.py", # passed
        "test_cumsum.py", # passed
        "test_gather.py", # passed
        "test_log1p.py", # passed
        "test_ones_like.py", # passed
        "test_sign.py", # passed
        "test_unbind.py", # passed
        "test_Tensor_erfinv_.py",
        "test_as_complex.py", # passed
        "test_diag.py",
        "test_gather_nd.py", # passed
        "test_log2.py", # passed
        "test_outer.py", # passed
        # "test_sin.py",
        "test_uniform.py", # passed
        "test_Tensor_exponential_.py",
        "test_as_real.py", # passed
        "test_diagflat.py", # passed
        "test_gcd.py", # passed
        "test_logical_and.py", # passed
        "test_poisson.py", # passed
        "test_sinh.py",
        "test_unique.py", # passed
        "test_Tensor_gcd.py", # passed
        "test_asin.py", # passed
        "test_diagonal.py", # passed
        "test_grad.py", # passed
        "test_logical_not.py", # passed
        "test_pow.py",
        "test_slice.py", # passed
        "test_unique_consecutive.py", # passed
        "test_Tensor_inner.py", # passed
        "test_assign.py", # passed
        "test_diff.py", # passed
        "test_greater_equal.py", # passed
        "test_logical_or.py", # passed
        "test_prod.py", # passed
        "test_sort.py", # passed
        "test_unsqueeze.py", # passed
        "test_Tensor_isclose.py", # passed
        "test_atan.py", # passed
        "test_digamma.py",
        "test_greater_than.py", # passed
        "test_logical_xor.py", # passed
        "test_put_along_axis.py", # passed
        "test_split.py", # passed
        "test_unsqueeze_.py", # passed
        "test_Tensor_lerp.py", # passed
        "test_atan2.py",
        "test_dist.py",
        "test_histogram.py",
        "test_logit.py",
        "test_put_along_axis_.py", # passed
        # "test_sqrt.py", # passed
        "test_unstack.py", # passed
        "test_Tensor_lerp_.py", # passed
        "test_bernoulli.py", # passed
        # "test_divide.py", # passed
        "test_imag.py",
        "test_logsumexp.py", # passed
        "test_quantile.py",
        "test_square.py", # passed
        "test_var.py",
        "test_Tensor_logit.py",
        "test_bincount.py", # hang
        "test_dot.py", # passed
        "test_in_dynamic_mode.py", # passed
        "test_masked_select.py", # passed
        "test_rad2deg.py",
        "test_squeeze.py", # passed
        # "test_where.py", # passed
        "test_Tensor_lu.py",
        "test_bitwise_and.py", # passed
        "test_dygraph_setitem.py", # passed
        "test_increment.py", # passed
        "test_matmul.py", # passed 
        "test_rand.py", # passed
        "test_squeeze_.py", # passed
        "test_zero_.py", # passed
        "test_Tensor_mode.py", # passed
        "test_bitwise_not.py", # passed
        "test_einsum.py",
        "test_index_sample.py", # passed
        "test_max.py", # passed
        "test_randint.py", # passed
        "test_stack.py", # passed
        "test_zeros.py",
        "test_Tensor_moveaxis.py", # passed
        "test_bitwise_or.py", # passed
        "test_empty.py", # passed
        "test_index_select.py",
        "test_maximum.py", # passed
        "test_randint_like.py", # passed
        "test_stanh.py", # passed
        "test_zeros_like.py", # passed
        "test_Tensor_nansum.py", # passed
        "test_bitwise_xor.py", # passed
        "test_empty_like.py", # passed
        "test_inner.py", # passed
        "test_mean.py", # passed
        "test_randn.py", # passed
        "test_std.py", # passed
        "test_Tensor_outer.py", # passed
        "test_bmm.py", # passed
        # "test_equal.py", # passed
        "test_inverse.py", # passed
        "test_median.py", # passed
        "test_randperm.py", # passed
        "test_strided_slice.py", # passed
        "test_Tensor_quantile.py",
        "test_broadcast_shape.py", # passed
        "test_equal_all.py", # passed
        "test_is_complex.py", # passed
        "test_meshgrid.py", # passed
        "test_rank.py", # passed
        "test_subtract.py", # passed
        "test_Tensor_rad2deg.py",
        "test_broadcast_tensors.py", # passed
        "test_erf.py", # passed
        "test_is_empty.py", # passed
        "test_min.py", # passed
        "test_real.py",
        "test_sum.py", # passed
        "test_Tensor_repeat_interleave.py", # passed
        "test_broadcast_tensors1.py", # passed
        "test_erfinv.py",
        "test_is_grad_enabled.py", # passed
        "test_minimum.py", # passed
        "test_reciprocal.py", # passed
        "test_summary.py", # passed
        "test_Tensor_rot90.py", # passed
        "test_broadcast_to.py", # passed
        "test_exp.py", # passed
        "test_is_tensor.py", # passed
        "test_mm.py",
        "test_renorm.py", # passed
        "test_take_along_axis.py", # passed
        "test_abs.py", # passed
        "test_cast.py", # passed
        "test_expand.py",
        "test_isclose.py", # passed
        "test_mode.py", # passed
        "test_repeat_interleave.py", # passed
        "test_tan.py", # passed
    ],

    "optimizer": [
        "test_adam.py", # passed
        "test_lr_cosineannealing.py", # passed
        "test_lr_multi_step_decay.py", # passed
        "test_lr_polynomial.py", # passed
        "test_momentum.py", # passed
        "test_MultiplicativeDecay.py", # passed
        "test_adamax.py", # passed
        "test_lr_exponential_decay.py", # passed
        "test_lr_natural_exp_decay.py", # passed
        "test_lr_reduce_on_plateau.py", # passed
        "test_rmsprop.py", # passed
        "test_adadelta.py",
        "test_adamw.py", # passed
        "test_lr_inverse_time_decay.py", # passed
        "test_lr_noam_decay.py", # passed
        "test_lr_step_decay.py", # passed
        "test_sgd.py", # passed
        "test_adagrad.py", # passed
        "test_lamb.py", # passed
        "test_lr_lambda.py", # passed
        "test_lr_piecewise.py", # passed
        "test_lr_warmup.py", # passed
        "test_sgd_params_group.py", # passed
    ],
    "distribution": [
        "test_AffineTransform.py", # passed
        "test_Dirichlet.py", # passed
        "test_IndependentTransform.py", # passed
        "test_PowerTransform.py", # passed
        "test_Beta.py", # passed
        "test_ExpTransform.py", # passed
        "test_kl_divergence.py", # passed
        "test_register_kl.py", # passed
        "test_AbsTransform.py", # passed
        "test_ChainTransform.py", # passed
        "test_Independent.py", # passed
        "test_Multinomial.py", # passed
        "test_ReshapeTransform.py", # passed
        "test_TransformedDistribution.py", # passed
        "test_SigmoidTransform.py", # passed
        "test_SoftmaxTransform.py", # passed
        "test_StickBreakingTransform.py", # passed
        "test_TanhTransform.py", # passed
        "test_StackTransform.py", # passed
    ],
    "utils": [
        "test_dlpack.py", # passed
    ],
}



def worker(fun):
    """worker"""
    while True:
        temp = taskQueue.get()
        fun(temp)
        taskQueue.task_done()


def threadPool(threadPoolNum):
    """threadPool"""
    threadPool = []
    for i in range(threadPoolNum):
        thread = threading.Thread(target=worker, args={doFun})
        thread.daemon = True
        threadPool.append(thread)
    return threadPool


def runCETest(params):
    """runCETest"""
    path = params[0]
    case = params[1]
    print("case: %s" % case)
    val = os.system("export FLAGS_call_stack_level= && cd %s && python -m pytest %s" % (path, case))
    retry_count = 0
    final_result = ""
    while val != 0:
        val = os.system("export FLAGS_call_stack_level=2 && cd %s && python -m pytest %s" % (path, case))
        retry_count = retry_count + 1
        if retry_count > 2:
            val = 0
            final_result = "Failed"
    if final_result == "Failed":
        failed_ce_case_list.append(case)
        os.system('echo "%s" >> %s/result.txt' % (case, path))


def doFun(params):
    """doFun"""
    runCETest(params)


def main(path):
    """
    1. run case
    """
    dirs = os.listdir(path)
    case_dir = path.split("/")[-1]
    os.system('echo "============ failed cases =============" >> %s/result.txt' % path)
    ignore_case_list = ignore_case_dir[case_dir]
    pool = threadPool(1)
    for i in range(pool.__len__()):
        pool[i].start()
    for case in dirs:
        if case.startswith("test") and case.endswith("py") and case not in ignore_case_list:
            params = [path, case]
            taskQueue.put(params)
    taskQueue.join()


if __name__ == "__main__":
    case_dir = sys.argv[1]
    pwd = os.getcwd()
    path = "%s/%s" % (pwd, case_dir)
    main(path)
    os.system('echo "total bugs: %s" >> %s/result.txt' % (len(failed_ce_case_list), path))
    sys.exit(len(failed_ce_case_list))
