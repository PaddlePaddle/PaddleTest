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
                "id": "5733be284776f41900661182",
                "title": "University_of_Notre_Dame",
                "context": "Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, "
                'is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, '
                "a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main "
                "drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.",
                "question": "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",
                "answers": ["Saint Bernadette Soubirous"],
                "answer_starts": [515],
                "is_impossible": False,
            }
        ),
    }
    return examples[mode]


class TestSQUAD(CpuCommonTest):
    """
    clue tnews case
    """

    def setUp(self):
        """
        check input params & datasets all flies
        """
        self.config["path_or_read_func"] = "squad"
        self.config["splits"] = ["train_v1", "dev_v1", "train_v2", "dev_v2"]

    def test_train_set(self):
        """
        check train.json length, label,text
        """
        expected_ds_num = 4
        expected_len = 87599
        expected_train = get_examples("train")
        ds = load_dataset(**self.config)
        self.check_output_equal(len(ds), expected_ds_num)
        self.check_output_equal(len(ds[0]), expected_len)

        self.check_output_equal(expected_train["id"], ds[0][0]["id"])
        self.check_output_equal(expected_train["title"], ds[0][0]["title"])
        self.check_output_equal(expected_train["context"], ds[0][0]["context"])
        self.check_output_equal(expected_train["answers"], ds[0][0]["answers"])
        self.check_output_equal(expected_train["answer_starts"], ds[0][0]["answer_starts"])
        self.check_output_equal(expected_train["is_impossible"], ds[0][0]["is_impossible"])


class TestSQUADNoSplitDataFiles(CpuCommonTest):
    """
    check no splits
    """

    def setUp(self):
        self.config["path_or_read_func"] = "squad"

    @util.assert_raises
    def test_no_split_datafiles(self):
        load_dataset(**self.config)


if __name__ == "__main__":
    unittest.main()
