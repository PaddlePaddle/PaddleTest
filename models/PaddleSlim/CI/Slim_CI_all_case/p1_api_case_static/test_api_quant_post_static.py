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
from paddleslim.quant import quant_post_static
from static_case import StaticCase
import numpy as np
from models import MobileNet

sys.path.append("../")
sys.path.append("../demo")


if paddle.is_compiled_with_cuda() is True:
    places = [paddle.CPUPlace(), paddle.CUDAPlace(0)]
else:
    # default
    places = [paddle.CPUPlace()]


class TestQuantAwareCase1(StaticCase):
    """
    Test  QuantAware
    """
    def test_accuracy(self):
        """
        paddleslim.quant.quant_post_static(executor,
        model_dir,quantize_model_path,
        batch_generator=None,sample_generator=None,
        model_filename=None,params_filename=None,
        save_model_filename='__model__',
        save_params_filename='__params__',
        batch_size=16, batch_nums=None,
        scope=None, algo='KL',
        quantizable_op_type=["conv2d","depthwise_conv2d","mul"],
        is_full_quantize=False,
        weight_bits=8,
        activation_bits=8,
        activation_quantize_type='range_abs_max',
        weight_quantize_type='channel_wise_abs_max',
        optimize_model=False)
        :return:
        """
        image = paddle.static.data(name="image", shape=[None, 1, 28, 28], dtype="float32")
        label = paddle.static.data(name="label", shape=[None, 1], dtype="int64")
        model = MobileNet()
        out = model.net(input=image, class_dim=10)
        cost = paddle.nn.functional.loss.cross_entropy(input=out, label=label)
        avg_cost = paddle.mean(x=cost)
        acc_top1 = paddle.metric.accuracy(input=out, label=label, k=1)
        acc_top5 = paddle.metric.accuracy(input=out, label=label, k=5)
        optimizer = paddle.optimizer.Momentum(
            momentum=0.9, learning_rate=0.01, weight_decay=paddle.regularizer.L2Decay(4e-5)
        )
        optimizer.minimize(avg_cost)
        main_prog = paddle.static.default_main_program()
        val_prog = main_prog.clone(for_test=True)

        place = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        exe.run(paddle.static.default_startup_program())

        def transform(x):
            return np.reshape(x, [1, 28, 28])

        train_dataset = paddle.vision.datasets.MNIST(mode="train", backend="cv2", transform=transform)
        test_dataset = paddle.vision.datasets.MNIST(mode="test", backend="cv2", transform=transform)
        train_loader = paddle.io.DataLoader(
            train_dataset, places=place, feed_list=[image, label], drop_last=True, return_list=False, batch_size=64
        )
        valid_loader = paddle.io.DataLoader(
            test_dataset, places=place, feed_list=[image, label], batch_size=64, return_list=False
        )

        def train(program):
            iter = 0
            for data in train_loader():
                cost, top1, top5 = exe.run(program, feed=data, fetch_list=[avg_cost, acc_top1, acc_top5])
                iter += 1
                if iter % 100 == 0:
                    print("train iter={}, avg loss {}, acc_top1 {}, acc_top5 {}".format(iter, cost, top1, top5))

        def test(program, outputs=[avg_cost, acc_top1, acc_top5]):
            iter = 0
            result = [[], [], []]
            for data in valid_loader():
                cost, top1, top5 = exe.run(program, feed=data, fetch_list=outputs)
                iter += 1
                if iter % 100 == 0:
                    print("eval iter={}, avg loss {}, acc_top1 {}, acc_top5 {}".format(iter, cost, top1, top5))
                result[0].append(cost)
                result[1].append(top1)
                result[2].append(top5)
            print(
                " avg loss {}, acc_top1 {}, acc_top5 {}".format(
                    np.mean(result[0]), np.mean(result[1]), np.mean(result[2])
                )
            )
            return np.mean(result[1]), np.mean(result[2])

        train(main_prog)
        top1_1, top5_1 = test(val_prog)
        paddle.fluid.io.save_inference_model(
            dirname="./test_quant_post",
            feeded_var_names=[image.name, label.name],
            target_vars=[avg_cost, acc_top1, acc_top5],
            main_program=val_prog,
            executor=exe,
            model_filename="model",
            params_filename="params",
        )

        quant_post_static(
            exe,
            model_dir="./test_quant_post",
            quantize_model_path="./test_quant_post_static_inference1",
            sample_generator=paddle.dataset.mnist.test(),
            model_filename="model",
            params_filename="params",
            batch_nums=10,
        )
        quant_post_prog, feed_target_names, fetch_targets = paddle.fluid.io.load_inference_model(
            dirname="./test_quant_post_static_inference1",
            executor=exe,
            model_filename="__model__",
            params_filename="__params__",
        )
        top1_2, top5_2 = test(quant_post_prog, fetch_targets)
        print("before quantization: top1: {}, top5: {}".format(top1_1, top5_1))
        print("after quantization: top1: {}, top5: {}".format(top1_2, top5_2))

    def test_quant_post_static2(self):
        """
        algo='KL',quantizable_op_type=["conv2d"],
        :return:
        """
        for place in places:
            exe = paddle.static.Executor(place)
            quant_post_static(
                exe,
                model_dir="./test_quant_post",
                quantize_model_path="./test_quant_post_static_inference2",
                sample_generator=paddle.dataset.mnist.test(),
                model_filename="model",
                params_filename="params",
                save_model_filename="__model__",
                save_params_filename="__params__",
                batch_size=8,
                batch_nums=1,
                algo="KL",
                quantizable_op_type=["conv2d"],
            )

    def test_quant_post_static3(self):
        """
        algo='abs_max',is_full_quantize=True,activation_quantize_type='moving_average_abs_max',
        :return:
        """
        for place in places:
            exe = paddle.static.Executor(place)
            quant_post_static(
                exe,
                model_dir="./test_quant_post",
                quantize_model_path="./test_quant_post_static_inference3",
                sample_generator=paddle.dataset.mnist.test(),
                model_filename="model",
                params_filename="params",
                save_model_filename="__model__",
                save_params_filename="__params__",
                batch_size=8,
                batch_nums=1,
                algo="abs_max",
                is_full_quantize=True,
                activation_quantize_type="moving_average_abs_max",
            )

    def test_quant_post_static4(self):
        """
        algo='KL',weight_quantize_type='abs_max',
        :return:
        """
        for place in places:
            exe = paddle.static.Executor(place)
            quant_post_static(
                exe,
                model_dir="./test_quant_post",
                quantize_model_path="./test_quant_post_static_inference4",
                sample_generator=paddle.dataset.mnist.test(),
                model_filename="model",
                params_filename="params",
                save_model_filename="__model__",
                save_params_filename="__params__",
                batch_size=8,
                batch_nums=1,
                algo="KL",
                quantizable_op_type=["conv2d"],
                weight_quantize_type="abs_max",
            )

    def test_quant_post_static5(self):
        """
        algo='abs_max',optimize_model=True,
        :return:
        """
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        quant_post_static(
            exe,
            model_dir="./test_quant_post",
            quantize_model_path="./test_quant_post_static_inference5",
            sample_generator=paddle.dataset.mnist.test(),
            model_filename="model",
            params_filename="params",
            save_model_filename="__model__",
            save_params_filename="__params__",
            batch_size=8,
            batch_nums=1,
            algo="abs_max",
            optimize_model=True,
            hist_percent=0.9999,
            bias_correction=True,
        )

    def test_quant_post_static6(self):
        """
        algo='hist',optimize_model=True,
        :return:
        """
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        quant_post_static(
            exe,
            model_dir="./test_quant_post",
            quantize_model_path="./test_quant_post_static_inference6",
            sample_generator=paddle.dataset.mnist.test(),
            model_filename="model",
            params_filename="params",
            save_model_filename="__model__",
            save_params_filename="__params__",
            batch_size=8,
            batch_nums=1,
            algo="hist",
            optimize_model=True,
            hist_percent=0.111,
            bias_correction=True,
        )

    def test_quant_post_static7(self):
        """
        algo='mse',optimize_model=True,
        :return:
        """
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        quant_post_static(
            exe,
            model_dir="./test_quant_post",
            quantize_model_path="./test_quant_post_static_inference7",
            sample_generator=paddle.dataset.mnist.test(),
            model_filename="model",
            params_filename="params",
            save_model_filename="__model__",
            save_params_filename="__params__",
            batch_size=8,
            batch_nums=1,
            algo="mse",
            optimize_model=True,
            hist_percent=0.111,
            bias_correction=True,
        )

    def test_quant_post_static8(self):
        """
        algo='avg',optimize_model=True,
        :return:
        """
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        quant_post_static(
            exe,
            model_dir="./test_quant_post",
            quantize_model_path="./test_quant_post_static_inference8",
            sample_generator=paddle.dataset.mnist.test(),
            model_filename="model",
            params_filename="params",
            save_model_filename="__model__",
            save_params_filename="__params__",
            batch_size=8,
            batch_nums=1,
            algo="avg",
            optimize_model=True,
            hist_percent=0.111,
            bias_correction=True,
        )


if __name__ == "__main__":
    unittest.main()
