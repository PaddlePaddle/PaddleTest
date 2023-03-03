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
        "train": ({"id": "sfq_9893", "title": "", "answers": ["hollinger internation"], "answer_starts": [199]}),
    }
    return examples[mode]


class TestTriviaqa(CpuCommonTest):
    """
    clue tnews case
    """

    def setUp(self):
        """
        check input params & datasets all flies
        """
        self.config["path_or_read_func"] = "triviaqa"
        self.config["splits"] = ["train", "dev"]

    def test_train_set(self):
        """
        check train.json length, label,text
        """
        expected_ds_num = 2
        expected_len = 106831
        expected_train = get_examples("train")
        ds = load_dataset(**self.config)
        self.check_output_equal(len(ds), expected_ds_num)
        self.check_output_equal(len(ds[0]), expected_len)

        # self.check_output_equal(expected_train['id'], ds[0][0]['id'])
        # self.check_output_equal(expected_train['title'], ds[0][0]['title'])
        # self.check_output_equal(expected_train['context'], ds[0][0]['context'])
        self.check_output_equal(expected_train["answers"], ds[0][0]["answers"])
        self.check_output_equal(expected_train["answer_starts"], ds[0][0]["answer_starts"])
        # self.check_output_equal(expected_train['is_impossible'], ds[0][0]['is_impossible'])


class TestTriviaqaNoSplitDataFiles(CpuCommonTest):
    """
    check no splits
    """

    def setUp(self):
        self.config["path_or_read_func"] = "triviaqa"

    @util.assert_raises
    def test_no_split_datafiles(self):
        load_dataset(**self.config)


if __name__ == "__main__":
    unittest.main()
