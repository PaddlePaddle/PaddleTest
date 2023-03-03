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
                "level": "hard",
                "sentence1": "七五期间开始,国家又投资将武汉市区的部分土堤改建为钢筋泥凝土防水墙",
                "sentence2": "八五期间会把剩下的土堤都改建完",
                "label": "neutral",
                "label0": "null",
                "label1": "null",
                "label2": "null",
                "label3": "null",
                "label4": "null",
                "genre": "news",
                "prem_id": "news_1553",
                "id": 0,
            }
        ),
        "dev": (
            {
                "level": "medium",
                "sentence1": "身上裹一件工厂发的棉大衣,手插在袖筒里",
                "sentence2": "身上至少一件衣服",
                "label": "entailment",
                "label0": "entailment",
                "label1": "entailment",
                "label2": "entailment",
                "label3": "entailment",
                "label4": "entailment",
                "genre": "lit",
                "prem_id": "lit_635",
                "id": 0,
            }
        ),
        "test": ({"sentence1": "来回一趟象我们两个人要两千五百块美金.", "sentence2": "我们有急事需要来回往返", "id": 0}),
    }
    return examples[mode]


class TestFewClueOCNLI(CpuCommonTest):
    """
    clue tnews case
    """

    def setUp(self):
        """
        check input params & datasets all flies
        """
        self.config["path_or_read_func"] = "fewclue"
        self.config["name"] = "ocnli"
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
        check train.json length, level,sentence1,sentence2,label,label0,label1,label2,label3,label4,genre,prem_id,id}
        """
        expected_ds_num = 15
        expected_len = 32
        expected_train = get_examples("train")
        ds = load_dataset(**self.config)
        self.check_output_equal(len(ds), expected_ds_num)
        self.check_output_equal(len(ds[0]), expected_len)

        self.check_output_equal(expected_train["level"], ds[0][0]["level"])
        self.check_output_equal(expected_train["sentence1"], ds[0][0]["sentence1"])
        self.check_output_equal(expected_train["sentence2"], ds[0][0]["sentence2"])
        self.check_output_equal(expected_train["label"], ds[0][0]["label"])
        self.check_output_equal(expected_train["genre"], ds[0][0]["genre"])
        self.check_output_equal(expected_train["prem_id"], ds[0][0]["prem_id"])


class TestFewClueOCNLINoSplitDataFiles(CpuCommonTest):
    """
    check no splits
    """

    def setUp(self):
        self.config["path_or_read_func"] = "fewclue"
        self.config["task_name"] = "ocnli"

    @util.assert_raises
    def test_no_split_datafiles(self):
        load_dataset(**self.config)


if __name__ == "__main__":
    unittest.main()
