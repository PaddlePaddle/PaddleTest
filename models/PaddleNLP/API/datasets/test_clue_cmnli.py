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


def get_examples(mode='train'):
    """
    dataset[0][0] examples
    """
    examples = {
        'train':
        ({"sentence1": "从概念上讲，奶油略读有两个基本维度-产品和地理。", "sentence2": "产品和地理位置是使奶油撇油起作用的原因。", "label": "neutral"}),
        'test':
        ({"sentence1": "新的权利已经足够好了", "sentence2": "每个人都很喜欢最新的福利", "label": "neutral"}),
        'dev':
        ({"id": 0, "sentence1": "最近，全世界都在看着最新的航天飞机进行处女航。", "sentence2": "全世界都在看着最近的航天飞机发射。"}),
    }
    return examples[mode]

class TestClueCMNLI(CpuCommonTest):
    """
    clue tnews case
    """
    def setUp(self):
        """
        check input params & datasets all flies
        """
        self.config['path_or_read_func'] = 'clue'
        self.config['name'] = 'cmnli'
        self.config['splits'] = ['train', 'dev','test']

    def test_train_set(self):
        """
        check train.json length, label,label_desc, sentence, keywords
        """
        expected_ds_num = 3
        expected_len = 391783
        expected_train= get_examples('train')
        ds = load_dataset(**self.config)
        self.check_output_equal(len(ds), expected_ds_num)
        self.check_output_equal(len(ds[0]), expected_len)
        expected_label =2
        if expected_train['label']=="contradiction":
            expected_label = 0 
        elif expected_train['label']=="entailment":
            expected_label = 1

        self.check_output_equal(expected_train['sentence1'], ds[0][0]['sentence1'])
        self.check_output_equal(expected_train['sentence2'], ds[0][0]['sentence2'])
        self.check_output_equal(expected_label, ds[0][0]['label'])


class TestClueCMNLINoSplitDataFiles(CpuCommonTest):
    """
    check no splits 
    """
    def setUp(self):
        self.config['path_or_read_func'] = 'clue'
        self.config['task_name'] = 'cmnli'

    @util.assert_raises
    def test_no_split_datafiles(self):
        load_dataset(**self.config)


if __name__ == "__main__":
    unittest.main()
