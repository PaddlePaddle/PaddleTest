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
        ({"target": {"span2_index": 30, "span1_index": 22, "span1_text": "树叶", "span2_text": "他"}, "idx": 0, "label": "false", "text": "为什么要出现一个身穿军装的高大男人？就像一片树叶飘入了树林，他走到了我的家人中间。"}),
        'dev':
        ({"target": {"span2_index": 35, "span1_index": 6, "span1_text": "洋人", "span2_text": "他们"}, "idx": 1, "label": "true", "text": "有些这样的“洋人”就站在大众之间，如同鹤立鸡群，毫不掩饰自己的优越感。他们排在非凡的甲菜盆后面，虽然人数寥寥无几，但却特别惹眼。"}),
        'test':
        ({"id": 0, "target": {"span1_index": 46, "span1_text": "曹操", "span2_index": 17, "span2_text": "他们"}, "text": "作为代价，这个兵种在不拔剑的时候，他们看起来就跟普通的精锐没什么区别，顺带从这里也可以看出来曹操给曹真补了这五千多锐士是怎么个想法，很明显曹操也不是好鸟。"}),
        'test1.0':
        ({"target": {"span2_index": 15, "span1_index": 0, "span1_text": "杨百顺", "span2_text": "他"}, "id": 1, "text": "杨百顺他爹是个卖豆腐的。别人叫他卖豆腐的老杨。"}),
    }
    return examples[mode]

class TestFewClueTNEWS(CpuCommonTest):
    """
    clue tnews case
    """
    def setUp(self):
        """
        check input params & datasets all flies
        """
        self.config['path_or_read_func'] = 'fewclue'
        self.config['name'] = 'cluewsc'
        self.config['splits'] = ['train_0', 'train_1', 'train_2', 'train_3', 'train_4', 'train_few_all', 'dev_0', 'dev_1', 'dev_2', 'dev_3', 'dev_4', 'dev_few_all', 'test', 'test_public']

    def test_train_set(self):
        """
        check train.json length, label,label_desc, sentence, keywords
        """
        expected_ds_num = 14
        expected_len = 32
        expected_train= get_examples('train')
        ds = load_dataset(**self.config)
        self.check_output_equal(len(ds), expected_ds_num)
        self.check_output_equal(len(ds[0]), expected_len)

        self.check_output_equal(expected_train['target']['span1_text'], ds[0][0]['target']['span1_text'])
        self.check_output_equal(expected_train['idx'], ds[0][0]['idx'])
        self.check_output_equal(expected_train['label'], 'false' if ds[0][0]['label'] else 'true')
        self.check_output_equal(expected_train['text'], ds[0][0]['text'])



class TestFewClueTNEWSNoSplitDataFiles(CpuCommonTest):
    """
    check no splits 
    """
    def setUp(self):
        self.config['path_or_read_func'] = 'fewclue'
        self.config['task_name'] = 'cluewsc'

    @util.assert_raises
    def test_no_split_datafiles(self):
        load_dataset(**self.config)


if __name__ == "__main__":
    unittest.main()
