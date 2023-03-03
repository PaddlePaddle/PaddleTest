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
                "id": 0,
                "question": "$recycle.bin文件夹可以删除么",
                "answer": "$RECYCLE.BIN 是 Win7、vista 的回收站，RECYCLER "
                "是 XP 的回收站，如果是 xp、win7双系统机器，或者 xp、vista 双系统机器，xp 系统也会有$RECYCLE.BIN，这是系统文件，不是病毒，不需要删除。",
                "labels": 1,
            }
        ),
    }
    return examples[mode]


class TestDireaderYesno(CpuCommonTest):
    """
    clue tnews case
    """

    def setUp(self):
        """
        check input params & datasets all flies
        """
        self.config["path_or_read_func"] = "dureader_yesno"
        self.config["splits"] = ["train", "dev", "test"]

    def test_train_set(self):
        """
        check train.json length, label,text
        """
        expected_ds_num = 3
        expected_len = 75391
        expected_train = get_examples("train")
        ds = load_dataset(**self.config)
        self.check_output_equal(len(ds), expected_ds_num)
        self.check_output_equal(len(ds[0]), expected_len)

        self.check_output_equal(expected_train["id"], ds[0][0]["id"])
        self.check_output_equal(expected_train["question"], ds[0][0]["question"])
        self.check_output_equal(expected_train["answer"], ds[0][0]["answer"])
        self.check_output_equal(expected_train["labels"], ds[0][0]["labels"])


class TestDureaderYesnoNoSplitDataFiles(CpuCommonTest):
    """
    check no splits
    """

    def setUp(self):
        self.config["path_or_read_func"] = "dureader_yesno"

    @util.assert_raises
    def test_no_split_datafiles(self):
        load_dataset(**self.config)


if __name__ == "__main__":
    unittest.main()
