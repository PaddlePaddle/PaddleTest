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
        ({"label": "108", "label_desc": "news_edu", "sentence": "上课时学生手机响个不停，老师一怒之下把手机摔了，家长拿发票让老师赔，大家怎么看待这种事？", "keywords": ""}),
        'dev':
        ({"label": "102", "label_desc": "news_entertainment", "sentence": "江疏影甜甜圈自拍，迷之角度竟这么好看，美吸引一切事物", "keywords": "江疏影,美少女,经纪人,甜甜圈"}),
        'test':
        ({"id": 0, "sentence": "A股：2个细分领域龙头个股值得股民关注", "keywords": ""}),
        'test1.0':
        ({"id": 0, "sentence": "在设计史上，每当相对稳定的发展时期，这种设计思想就会成为主导", "keywords": "民族性,设计思想,继承型设计,复古主义,服装史"}),
    }
    return examples[mode]

class TestClueTNEWS(CpuCommonTest):
    """
    clue tnews case
    """
    def setUp(self):
        """
        check input params & datasets all flies
        """
        self.config['path_or_read_func'] = 'clue'
        self.config['name'] = 'tnews'
        self.config['splits'] = ['train', 'dev','test','test1.0','labels']

    def test_train_set(self):
        """
        check train.json length, label,label_desc, sentence, keywords
        """
        expected_ds_num = 5
        expected_len = 53360
        expected_train= get_examples('train')
        ds = load_dataset(**self.config)
        self.check_output_equal(len(ds), expected_ds_num)
        self.check_output_equal(len(ds[0]), expected_len)

        self.check_output_equal(int(expected_train['label'])-101, ds[0][0]['label'])
        self.check_output_equal(expected_train['label_desc'], ds[0][0]['label_desc'])
        self.check_output_equal(expected_train['sentence'], ds[0][0]['sentence'])
        self.check_output_equal(expected_train['keywords'], ds[0][0]['keywords'])



class TestClueTNEWSNoSplitDataFiles(CpuCommonTest):
    """
    check no splits 
    """
    def setUp(self):
        self.config['path_or_read_func'] = 'clue'
        self.config['task_name'] = 'tnews'

    @util.assert_raises
    def test_no_split_datafiles(self):
        load_dataset(**self.config)


if __name__ == "__main__":
    unittest.main()
