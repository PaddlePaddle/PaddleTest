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
# print(os.pardir)

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
                "results_recognition": [
                    "呃呃呃，最后一站，然后回到了大本营北京。",
                    "嗯呃，最后一站，然后回到了大本营北京。",
                    "呃呃，最后一站，然后回到了大本营北京。",
                    "呃呃，最后一站，然后回到了大本营北京。",
                    "啊那呃最后一站，然后回到了大本营北京。",
                ],
                "origin_result": {
                    "corpus_no": 6690206793487570294,
                    "err_no": 0,
                    "result": {
                        "word": [
                            "呃呃呃，最后一站，然后回到了大本营北京。",
                            "嗯呃，最后一站，然后回到了大本营北京。",
                            "呃呃，最后一站，然后回到了大本营北京。",
                            "呃呃，最后一站，然后回到了大本营北京。",
                            "啊那呃最后一站，然后回到了大本营北京。",
                        ]
                    },
                    "sn": "83742761-2AD6-4C0C-ABB8-5DE9071C303C",
                },
                "sn_start_time": "00:00.320",
                "sn_end_time": "00:05.105",
            }
        ),
    }
    return examples[mode]


class TestBSTC_ASR(CpuCommonTest):
    """
    clue tnews case
    """

    def setUp(self):
        """
        check input params & datasets all flies
        """
        self.config["path_or_read_func"] = "bstc"
        self.config["name"] = "asr"
        self.config["splits"] = ["train", "dev"]

    def test_train_set(self):
        """
        check train.json length,
        """
        expected_ds_num = 2
        expected_len = 50031
        expected_train = get_examples("train")
        ds = load_dataset(**self.config)
        self.check_output_equal(len(ds), expected_ds_num)
        self.check_output_equal(len(ds[0]), expected_len)

        self.check_output_equal(expected_train["results_recognition"], ds[0][0]["results_recognition"])
        self.check_output_equal(expected_train["origin_result"]["corpus_no"], ds[0][0]["origin_result"]["corpus_no"])
        self.check_output_equal(expected_train["origin_result"]["sn"], ds[0][0]["origin_result"]["sn"])
        self.check_output_equal(expected_train["sn_start_time"], ds[0][0]["sn_start_time"])


class TestBSTC_ASRNoSplitDataFiles(CpuCommonTest):
    """
    check no splits
    """

    def setUp(self):
        self.config["path_or_read_func"] = "bstc"
        self.config["task_name"] = "asr"

    @util.assert_raises
    def test_no_split_datafiles(self):
        load_dataset(**self.config)


if __name__ == "__main__":
    unittest.main()
