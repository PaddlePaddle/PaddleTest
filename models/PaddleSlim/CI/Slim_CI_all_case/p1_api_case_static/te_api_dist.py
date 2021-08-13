# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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
import sys
import unittest
import paddle
from paddleslim.dist import merge, fsp_loss
from layers import conv_bn_layer
from static_case import StaticCase

sys.path.append("../")

if paddle.is_compiled_with_cuda() is True:
    # places = [paddle.CPUPlace(), paddle.CUDAPlace(0)]
    places = [paddle.CPUPlace()]

else:
    # default
    places = [paddle.CPUPlace()]


class TestMerge(StaticCase):
    """
    test paddleslim.dist.merge
    """
    def test_merge_1(self):
        """
        test api :paddleslim.dist.merge
        :return:
        """
        for place in places:
            student_main = paddle.static.Program()
            student_startup = paddle.static.Program()
            with paddle.static.program_guard(student_main, student_startup):
                input = paddle.static.data(name="image", shape=[None, 3, 224, 224])
                conv1 = conv_bn_layer(input, 8, 3, "conv1")
                conv2 = conv_bn_layer(conv1, 8, 3, "conv2")
                conv1 + conv2
            student_ops = []
            for block in student_main.blocks:
                for op in block.ops:
                    student_ops.append(op)

            teacher_main = paddle.static.Program()
            teacher_startup = paddle.static.Program()
            with paddle.static.program_guard(teacher_main, teacher_startup):
                input = paddle.static.data(name="image", shape=[None, 3, 224, 224])
                conv1 = conv_bn_layer(input, 8, 3, "conv1")
                conv2 = conv_bn_layer(conv1, 8, 3, "conv2")
                sum1 = conv1 + conv2
                conv3 = conv_bn_layer(sum1, 8, 3, "conv3")
                conv4 = conv_bn_layer(conv3, 8, 3, "conv4")
                sum2 = conv4 + sum1
                conv5 = conv_bn_layer(sum2, 8, 3, "conv5")
                conv_bn_layer(conv5, 8, 3, "conv6")
            teacher_ops = []
            for block in teacher_main.blocks:
                for op in block.ops:
                    teacher_ops.append(op)

            data_name_map = {"image": "image"}
            merge(teacher_main, student_main, data_name_map, place, name_prefix="teacher_")
            merged_ops = []
            for block in student_main.blocks:
                for op in block.ops:
                    merged_ops.append(op)
            self.assertTrue(len(student_ops) + len(teacher_ops) == len(merged_ops))

    def test_fsp_loss_1(self):
        """
        paddleslim.dist.fsp_loss
        :return:
        """
        for place in places:
            input = paddle.static.data(name="image", shape=[None, 3, 224, 224])
            conv1 = conv_bn_layer(input, 8, 3, "conv1")
            conv2 = conv_bn_layer(conv1, 8, 3, "conv2")
            conv1 + conv2

            teacher_main = paddle.static.Program()
            teacher_startup = paddle.static.Program()
            with paddle.static.program_guard(teacher_main, teacher_startup):
                input = paddle.static.data(name="image", shape=[None, 3, 224, 224])
                conv1 = conv_bn_layer(input, 8, 3, "conv1")
                conv2 = conv_bn_layer(conv1, 8, 3, "conv2")
                sum1 = conv1 + conv2
                conv3 = conv_bn_layer(sum1, 8, 3, "conv3")
                conv4 = conv_bn_layer(conv3, 8, 3, "conv4")
                sum2 = conv4 + sum1
                conv5 = conv_bn_layer(sum2, 8, 3, "conv5")
                conv_bn_layer(conv5, 8, 3, "conv6")

            data_name_map = {"image": "image"}
            merge(teacher_main, paddle.static.default_main_program(), data_name_map, place)
            merged_ops = []
            for block in paddle.static.default_main_program().blocks:
                for op in block.ops:
                    merged_ops.append(op.type)
            fsp_loss(
                "teacher_conv5_bn_output.tmp_2",
                "teacher_conv6_bn_output.tmp_2",
                "conv1_bn_output.tmp_2",
                "conv2_bn_output.tmp_2",
            )
            loss_ops = []
            for block in paddle.static.default_main_program().blocks:
                for op in block.ops:
                    loss_ops.append(op.type)
            self.assertTrue(set(merged_ops).difference(set(loss_ops)) == set())
            self.assertTrue(
                set(loss_ops).difference(set(merged_ops)) == {"elementwise_sub", "reduce_mean", "square", "fsp"}
            )


if __name__ == "__main__":
    unittest.main()
