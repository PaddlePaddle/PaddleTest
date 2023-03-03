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
    examples = {"train": ({"id": 0, "sentence1": "叫爸爸叫一声我听听", "sentence2": "那你叫我一声爸爸", "label": "1"})}
    return examples[mode]


class TestFewClueBUSTM(CpuCommonTest):
    """
    clue tnews case
    """

    def setUp(self):
        """
        check input params & datasets all flies
        """
        self.config["path_or_read_func"] = "fewclue"
        self.config["name"] = "bustm"
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
        check train.json length, label,label_des, sentence,id
        """
        expected_ds_num = 15
        expected_len = 32
        expected_train = get_examples("train")
        ds = load_dataset(**self.config)
        self.check_output_equal(len(ds), expected_ds_num)
        self.check_output_equal(len(ds[0]), expected_len)

        self.check_output_equal(expected_train["label"], ds[0][0]["label"])
        self.check_output_equal(expected_train["sentence1"], ds[0][0]["sentence1"])
        self.check_output_equal(expected_train["sentence2"], ds[0][0]["sentence2"])
        self.check_output_equal(expected_train["id"], ds[0][0]["id"])


class TestFewClueBUSTMNoSplitDataFiles(CpuCommonTest):
    """
    check no splits
    """

    def setUp(self):
        self.config["path_or_read_func"] = "fewclue"
        self.config["task_name"] = "bustm"

    @util.assert_raises
    def test_no_split_datafiles(self):
        load_dataset(**self.config)


if __name__ == "__main__":
    unittest.main()
