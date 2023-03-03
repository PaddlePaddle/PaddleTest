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
        ({'context': ['男：你今天晚上有时间吗?我们一起去看电影吧?', '女：你喜欢恐怖片和爱情片，但是我喜欢喜剧片，科幻片一般。所以……'], 'question': '女的最喜欢哪种电影?', 'choice': ['恐怖片', '爱情片', '喜剧片', '科幻片'], 'answer': '喜剧片', 'label': 2}),
    }
    return examples[mode]

class TestC3(CpuCommonTest):
    """
    clue tnews case
    """
    def setUp(self):
        """
        check input params & datasets all flies
        """
        self.config['path_or_read_func'] = 'c3'
        self.config['splits'] = ['train','dev','test']

    def test_train_set(self):
        """
        check train.json length, label,text
        """
        expected_ds_num = 3
        expected_len = 11869
        expected_train= get_examples('train')
        ds = load_dataset(**self.config)
        self.check_output_equal(len(ds), expected_ds_num)
        self.check_output_equal(len(ds[0]), expected_len)

        self.check_output_equal(expected_train['question'], ds[0][0]['question'])
        self.check_output_equal(expected_train['choice'], ds[0][0]['choice'])
        self.check_output_equal(expected_train['context'], ds[0][0]['context'])
        self.check_output_equal(expected_train['answer'], ds[0][0]['answer'])
        self.check_output_equal(expected_train['label'], ds[0][0]['label'])


class TestC3NoSplitDataFiles(CpuCommonTest):
    """
    check no splits 
    """
    def setUp(self):
        self.config['path_or_read_func'] = 'c3'

    @util.assert_raises
    def test_no_split_datafiles(self):
        load_dataset(**self.config)


if __name__ == "__main__":
    unittest.main()
