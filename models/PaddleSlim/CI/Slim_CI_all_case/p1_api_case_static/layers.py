# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
@Desc:
@File:
@Author:
"""
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr


def conv_bn_layer(input, num_filters, filter_size, name, stride=1, groups=1, act=None, bias=False, use_cudnn=True):
    """
    construct conv and batch_norm layer
    """
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=(filter_size - 1) // 2,
        groups=groups,
        act=None,
        param_attr=ParamAttr(name=name + "_weights"),
        bias_attr=bias,
        name=name + "_out",
        use_cudnn=use_cudnn,
    )
    bn_name = name + "_bn"
    return fluid.layers.batch_norm(
        input=conv,
        act=act,
        name=bn_name + "_output",
        param_attr=ParamAttr(name=bn_name + "_scale"),
        bias_attr=ParamAttr(bn_name + "_offset"),
        moving_mean_name=bn_name + "_mean",
        moving_variance_name=bn_name + "_variance",
    )
