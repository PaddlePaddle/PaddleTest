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
                "sentence1": 'Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his evidence .',
                "sentence2": 'Referring to him as only " the witness " , Amrozi accused his brother of deliberately distorting his evidence .',
                "labels": 1,
            }
        ),
        "dev": (
            {
                "sentence1": "He said the foodservice pie business doesn 't fit the company 's long-term growth strategy .",
                "sentence2": '" The foodservice pie business does not fit our long-term growth strategy .',
                "labels": 1,
            }
        ),
        "test": (
            {
                "sentence1": "PCCW 's chief operating officer , Mike Butcher , and Alex Arena , the chief financial officer , will report directly to Mr So .",
                "sentence2": "Current Chief Operating Officer Mike Butcher and Group Chief Financial Officer Alex Arena will report to So .",
            }
        ),
    }
    return examples[mode]


class TestGlueMRPC(CpuCommonTest):
    """
    clue tnews case
    """

    def setUp(self):
        """
        check input params & datasets all flies
        """
        self.config["path_or_read_func"] = "glue"
        self.config["name"] = "mrpc"
        self.config["splits"] = ["train", "dev", "test"]

    def test_train_set(self):
        """
        check train.json length, label,sentence
        """
        expected_ds_num = 3
        expected_len = 3668
        expected_train = get_examples("train")
        ds = load_dataset(**self.config)
        self.check_output_equal(len(ds), expected_ds_num)
        self.check_output_equal(len(ds[0]), expected_len)

        self.check_output_equal(expected_train["sentence1"], ds[0][0]["sentence1"])
        self.check_output_equal(expected_train["sentence2"], ds[0][0]["sentence2"])
        self.check_output_equal(expected_train["labels"], ds[0][0]["labels"])


class TestClueMRPCNoSplitDataFiles(CpuCommonTest):
    """
    check no splits
    """

    def setUp(self):
        self.config["path_or_read_func"] = "glue"
        self.config["task_name"] = "mrpc"

    @util.assert_raises
    def test_no_split_datafiles(self):
        load_dataset(**self.config)


if __name__ == "__main__":
    unittest.main()
