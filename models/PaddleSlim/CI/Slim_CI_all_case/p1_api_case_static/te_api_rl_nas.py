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
from static_case import StaticCase
from paddleslim.nas import RLNAS

sys.path.append("../")


class TestRLNAS(StaticCase):
    """
    Test classpaddleslim.nas.RLNAS(key,...)
    """

    def test_RLNAS1(self):
        """
        classpaddleslim.nas.RLNAS(key,
        configs,
        use_gpu=False,
        server_addr=("", 8881),
        is_server=True,
        is_sync=False,
        save_controller=None,
        load_controller=None, **kwargs)
        :return:
        """
        port = 8773
        # config = [('MobileNetV2BlockSpace', {'block_mask': [0]})]
        config = [("ResNetBlockSpace2", {"block_mask": [0]})]
        rlnas = RLNAS(
            key="lstm",
            configs=config,
            server_addr=("", port),
            is_sync=False,
            controller_batch_size=1,
            lstm_num_layers=1,
            hidden_size=10,
            temperature=1.0,
            save_controller=False,
        )
        input = paddle.static.data(name="input", shape=[None, 3, 32, 32], dtype="float32")
        archs = rlnas.next_archs(1)[0]
        for arch in archs:
            output = arch(input)
            input = output
        print(output)

    def test_RLNAS2(self):
        """
        is_server=False,is_sync=True,
        :return:
        """
        config = [("MobileNetV2BlockSpace", {"block_mask": [0]})]
        RLNAS(
            key="lstm",
            configs=config,
            use_gpu=False,
            server_addr=("", 8774),
            is_server=True,
            is_sync=False,
            save_controller="./rlnas_controller1",
            load_controller=None,
            controller_batch_size=1,
            lstm_num_layers=1,
            hidden_size=10,
            temperature=1.0,
        )

        RLNAS(
            key="lstm",
            configs=config,
            use_gpu=False,
            server_addr=("172.0.0.1", 8774),
            is_server=False,
            is_sync=False,
            save_controller="./rlnas_controller2",
            load_controller="./rlnas_controller1",
            controller_batch_size=1,
            lstm_num_layers=1,
            hidden_size=10,
            temperature=1.0,
        )

    def te_RLNAS3(self):
        """
        key='ddpg'(need parl),is_server=False,is_sync=True,
        :return:
        """
        config = [("MobileNetV2BlockSpace", {"block_mask": [0]})]
        RLNAS(
            key="ddpg",
            configs=config,
            use_gpu=False,
            server_addr=("", 8776),
            is_server=True,
            is_sync=True,
            save_controller="./rlnas_controller1",
            load_controller=None,
        )

        RLNAS(
            key="ddpg",
            configs=config,
            use_gpu=False,
            server_addr=("localhost", 8776),
            is_server=False,
            is_sync=True,
            save_controller="./rlnas_controller2",
            load_controller="./rlnas_controller1",
        )


if __name__ == "__main__":
    unittest.main()
