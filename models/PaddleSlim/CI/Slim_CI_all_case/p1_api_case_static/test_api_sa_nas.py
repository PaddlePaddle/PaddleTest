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
from static_case import StaticCase
from paddleslim.nas import SANAS

sys.path.append("../")


class TestSANAS(StaticCase):
    """
    Test SANAS
    """

    def test_SANAS1(self):
        """
        classpaddleslim.nas.SANAS(configs,
        server_addr=("", 8881),
        init_temperature=None,
        reduce_rate=0.85,
        init_tokens=None,
        search_steps=300,
        save_checkpoint='./nas_checkpoint',
        load_checkpoint=None,
        is_server=True)
        :return:
        """
        port = 8773
        config = [("MobileNetV2BlockSpace", {"block_mask": [0]})]
        SANAS(
            configs=config,
            server_addr=("", port),
            init_temperature=0.7,
            reduce_rate=0.8,
            init_tokens=None,
            search_steps=1,
            save_checkpoint="./nas_checkpoint1",
            load_checkpoint=None,
            is_server=True,
        )

    def test_SANAS2(self):
        """
        is_server=False
        :return:
        """
        port = 8774
        config = [("MobileNetV2BlockSpace", {"block_mask": [0]})]
        SANAS(
            configs=config,
            server_addr=("", port),
            init_temperature=0.7,
            reduce_rate=0.8,
            init_tokens=None,
            search_steps=1,
            save_checkpoint="./nas_checkpoint1",
            load_checkpoint=None,
            is_server=True,
        )

        SANAS(
            configs=config,
            server_addr=("172.0.0.1", port),
            init_temperature=0.7,
            reduce_rate=0.8,
            init_tokens=None,
            search_steps=1,
            save_checkpoint="./nas_checkpoint2",
            load_checkpoint="./nas_checkpoint1",
            is_server=False,
        )


if __name__ == "__main__":
    unittest.main()
