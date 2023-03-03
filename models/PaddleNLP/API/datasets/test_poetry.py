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
                "tokens": "画\x02精\x02禅\x02室\x02冷\x02，\x02方\x02暑\x02久\x02徘\x02徊\x02。",
                "labels": "不\x02尽\x02林\x02端\x02雪\x02，\x02长\x02青\x02石\x02上\x02苔\x02。\x02心\x02闲\x02对\x02岩\x02岫\x02，\x02目\x02浄\x02失\x02尘\x02埃\x02。\x02坐\x02久\x02清\x02风\x02至\x02，\x02疑\x02从\x02翠\x02涧\x02来\x02。",
            }
        ),
    }
    return examples[mode]


class TestPOETRY(CpuCommonTest):
    """
    clue tnews case
    """

    def setUp(self):
        """
        check input params & datasets all flies
        """
        self.config["path_or_read_func"] = "poetry"
        self.config["splits"] = ["train", "dev", "test"]

    def test_train_set(self):
        """
        check train.json length, label,label_desc, sentence, keywords
        """
        expected_ds_num = 3
        expected_len = 294598
        expected_train = get_examples("train")
        ds = load_dataset(**self.config)
        self.check_output_equal(len(ds), expected_ds_num)
        self.check_output_equal(len(ds[0]), expected_len)

        self.check_output_equal(expected_train["tokens"], ds[0][0]["tokens"])
        self.check_output_equal(expected_train["labels"], ds[0][0]["labels"])


class TestPOETRYNoSplitDataFiles(CpuCommonTest):
    """
    check no splits
    """

    def setUp(self):
        self.config["path_or_read_func"] = "poetry"

    @util.assert_raises
    def test_no_split_datafiles(self):
        load_dataset(**self.config)


if __name__ == "__main__":
    unittest.main()
