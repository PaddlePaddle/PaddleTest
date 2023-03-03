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
                "sentence1": "I stuck a pin through a carrot. When I pulled the pin out, it had a hole.",
                "sentence2": "The carrot had a hole.",
                "labels": 1,
            }
        ),
        "dev": (
            {
                "sentence1": "The drain is clogged with hair. It has to be cleaned.",
                "sentence2": "The hair has to be cleaned.",
                "labels": 0,
            }
        ),
        "test": (
            {
                "sentence1": "Maude and Dora had seen the trains rushing across the prairie, with long, rolling puffs of black smoke streaming back from the engine. Their roars and their wild, clear whistles could be heard from far away. Horses ran away when they came in sight.",
                "sentence2": "Horses ran away when Maude and Dora came in sight.",
            }
        ),
    }
    return examples[mode]


class TestGlueWNLI(CpuCommonTest):
    """
    clue tnews case
    """

    def setUp(self):
        """
        check input params & datasets all flies
        """
        self.config["path_or_read_func"] = "glue"
        self.config["name"] = "wnli"
        self.config["splits"] = ["train", "dev", "test"]

    def test_train_set(self):
        """
        check train.json length, label,sentence
        """
        expected_ds_num = 3
        expected_len = 635
        expected_train = get_examples("train")
        ds = load_dataset(**self.config)
        self.check_output_equal(len(ds), expected_ds_num)
        self.check_output_equal(len(ds[0]), expected_len)

        self.check_output_equal(expected_train["sentence1"], ds[0][0]["sentence1"])
        self.check_output_equal(expected_train["sentence2"], ds[0][0]["sentence2"])
        self.check_output_equal(expected_train["labels"], ds[0][0]["labels"])


class TestClueWNLINoSplitDataFiles(CpuCommonTest):
    """
    check no splits
    """

    def setUp(self):
        self.config["path_or_read_func"] = "glue"
        self.config["task_name"] = "wnli"

    @util.assert_raises
    def test_no_split_datafiles(self):
        load_dataset(**self.config)


if __name__ == "__main__":
    unittest.main()
