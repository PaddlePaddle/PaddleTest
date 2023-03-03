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
import numpy as np
import os
import unittest
from paddlenlp.datasets import load_dataset

import sys

sys.path.append(os.pardir)

from common_test import CpuCommonTest
import util
import unittest


def get_examples(mode="train"):
    """
    dataset[0][0] examples
    """
    examples = {
        "train": (
            {
                "id": 242,
                "candidates": ["风云人物", "气势汹汹", "予取予求", "乘龙佳婿", "正中下怀", "天方夜谭", "心如刀割"],
                "content": "据俄罗斯卫星通讯社3月15日报道,新八国联军#idiom#逼近附近海域,但是军舰却遭岸舰导弹锁定,英承认今非昔比。 最近一段时间,北约多个国家开始频繁进行军事演习,来对其他国家进行威慑。3月12日当天,英国出动了兰开斯特号、威斯敏斯特号...",
                "answer": 1,
            }
        )
    }
    return examples[mode]


class TestFewClueChid(CpuCommonTest):
    """
    clue tnews case
    """

    def setUp(self):
        """
        check input params & datasets all flies
        """
        self.config["path_or_read_func"] = "fewclue"
        self.config["name"] = "chid"
        self.config["splits"] = [
            "train_0",
            "train_1",
            "train_2",
            "train_3",
            "train_4",
            "train_few_all",
            "dev_0",
            "dev_1",
            "dev_2",
            "dev_3",
            "dev_4",
            "dev_few_all",
            "unlabeled",
            "test",
            "test_public",
        ]

    def test_train_set(self):
        """
        check train.json length, id, candidates, content, answer
        """
        expected_ds_num = 15
        expected_len = 42
        expected_train = get_examples("train")
        ds = load_dataset(**self.config)
        self.check_output_equal(len(ds), expected_ds_num)
        self.check_output_equal(len(ds[0]), expected_len)

        self.check_output_equal(int(expected_train["answer"]), ds[0][0]["answer"])
        self.check_output_equal(expected_train["candidates"], ds[0][0]["candidates"])
        self.check_output_equal(expected_train["content"], ds[0][0]["content"])
        self.check_output_equal(expected_train["id"], ds[0][0]["id"])


class TestFewclueChidNoSplitDataFiles(CpuCommonTest):
    """
    check no splits
    """

    def setUp(self):
        self.config["path_or_read_func"] = "fewclue"
        self.config["task_name"] = "chid"

    @util.assert_raises
    def test_no_split_datafiles(self):
        load_dataset(**self.config)


if __name__ == "__main__":
    unittest.main()
