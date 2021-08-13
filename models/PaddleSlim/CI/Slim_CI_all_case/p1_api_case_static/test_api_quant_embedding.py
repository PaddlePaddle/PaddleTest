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
import sys
import unittest
import paddle
import paddleslim.quant as quant
from static_case import StaticCase
sys.path.append("../")

if paddle.is_compiled_with_cuda() is True:
    places = [paddle.CPUPlace(), paddle.CUDAPlace(0)]
else:
    # default
    places = [paddle.CPUPlace()]


class TestQuantEmbedding(StaticCase):
    """
    Test paddleslim.quant.quant_embedding
    """
    def test_quant_embedding(self):
        """
        paddleslim.quant.quant_embedding(program, place, config=None, scope=None)
        :return:
        """
        train_program = paddle.static.Program()
        with paddle.static.program_guard(train_program):
            input_word = paddle.static.data(name="input_word", shape=[None, 1], dtype="int64")
            param_attr = paddle.ParamAttr(name="emb", initializer=paddle.nn.initializer.Uniform(-0.005, 0.005))
            weight = train_program.global_block().create_parameter((100, 128), attr=param_attr, dtype="float32")

            paddle.nn.functional.embedding(x=input_word, weight=weight, sparse=True)

        infer_program = train_program.clone(for_test=True)

        place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() else paddle.static.CPUPlace()
        exe = paddle.static.Executor(place)
        exe.run(paddle.static.default_startup_program())

        quant.quant_embedding(infer_program, place)

    def test_quant_embedding1(self):
        """
        paddleslim.quant.quant_embedding(program, place, config=None, scope=None)
        :return:
        """
        for place in places:
            train_program = paddle.static.Program()
            with paddle.static.program_guard(train_program):
                input_word = paddle.static.data(name="input_word", shape=[None, 1], dtype="int64")
                param_attr = paddle.ParamAttr(name="emb", initializer=paddle.nn.initializer.Uniform(-0.005, 0.005))
                weight = train_program.global_block().create_parameter((100, 128), attr=param_attr, dtype="float32")

                paddle.nn.functional.embedding(x=input_word, weight=weight, sparse=True)

            infer_program = train_program.clone(for_test=True)

            exe = paddle.static.Executor(place)
            exe.run(paddle.static.default_startup_program())
            config = {"quantize_op_types": ["lookup_table"], "lookup_table": {"quantize_type": "abs_max"}}
            quant.quant_embedding(infer_program, config=config, place=place)

    def test_quant_embedding2(self):
        """
        paddleslim.quant.quant_embedding(program, place, config=None, scope=None)
        :return:
        """
        for place in places:
            train_program = paddle.static.Program()
            with paddle.static.program_guard(train_program):
                input_word = paddle.static.data(name="input_word", shape=[None, 1], dtype="int64")
                param_attr = paddle.ParamAttr(name="emb", initializer=paddle.nn.initializer.Uniform(-0.005, 0.005))
                weight = train_program.global_block().create_parameter((100, 128), attr=param_attr, dtype="float32")

                paddle.nn.functional.embedding(x=input_word, weight=weight, sparse=True)

            infer_program = train_program.clone(for_test=True)

            exe = paddle.static.Executor(place)
            exe.run(paddle.static.default_startup_program())
            config = {
                "quantize_op_types": ["lookup_table", "fused_embedding_seq_pool", "pyramid_hash"],
                "lookup_table": {"quantize_type": "log"},
            }
            quant.quant_embedding(infer_program, config=config, place=place)


if __name__ == "__main__":
    unittest.main()
