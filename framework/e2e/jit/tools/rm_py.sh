#!/usr/bin/env bash

ignore="test_randint_base.py \
test_rand_0.py \
test_tanh__0.py \
test_multinomial_2.py \
test_bernoulli_base.py \
test_rand_base.py \
test_randint_like_base.py \
test_randint_0.py \
test_multinomial_1.py \
test_normal_base.py \
test_squeeze__0.py \
test_median_4.py \
test_SimpleRNN_5.py \
test_scale_2.py \
test_standard_normal_base.py \
test_randn_base.py \
test_median_3.py \
test_erfinv__base.py \
test_scale_5.py \
test_expand_as_0.py \
test_unsqueeze__base.py \
test_tanh__base.py \
test_squeeze__2.py \
test_uniform_base.py \
test_expand_as_base.py \
test_uniform_1.py \
test_poisson_base.py \
test_randint_like_1.py \
test_normal_1.py \
test_multinomial_base.py \
test_scale_4.py \
test_ones_2.py \
test_MaxUnPool2D_3.py \
test_multinomial_0.py \
test_conv3d_transpose_5.py \
test_MaxUnPool2D_0.py \
test_kthvalue_1.py \
test_uniform_0.py \
test_MaxUnPool2D_2.py \
test_median_1.py \
test_normal_0.py \
test_randint_1.py \
test_put_along_axis_0.py \
test_put_along_axis_base.py \
test_put_along_axis_1.py \
test_randint_like_0.py \
test_randperm_base.py \
test_median_base.py \
test_squeeze__1.py \
test_MaxUnPool2D_1.py \
test_scale_3.py \
test_tanh__1.py \
test_unsqueeze__0.py \
test_MaxUnPool2D_base.py
"

for file in ${ignore}
do
echo ${file} is coming!!!
rm -rf ${file}
done
