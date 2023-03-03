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
        ({'id': '0a25cb4bc1ab6f474c699884e04601e4', 'title': '', 'context': '第35集雪见缓缓张开眼睛，景天又惊又喜之际，长卿和紫萱的仙船驶至，见众人无恙，'
        '也十分高兴。众人登船，用尽合力把自身的真气和水分输给她。雪见终于醒过来了，但却一脸木然，全无反应。众人向常胤求助，却发现人世界竟没有雪见的身世纪录。长卿询问清微的身世，'
        '清微语带双关说一切上了天界便有答案。长卿驾驶仙船，众人决定立马动身，往天界而去。众人来到一荒山，长卿指出，魔界和天界相连。由魔界进入通过神魔之井，便可登天。众人至魔界入口，'
        '仿若一黑色的蝙蝠洞，但始终无法进入。后来花楹发现只要有翅膀便能飞入。于是景天等人打下许多乌鸦，模仿重楼的翅膀，制作数对翅膀状巨物。刚佩戴在身，便被吸入洞口。众人摔落在地，'
        '抬头发现魔界守卫。景天和众魔套交情，自称和魔尊重楼相熟，众魔不理，打了起来。', 'question': '仙剑奇侠传3第几集上天界', 'answers': ['第35集'], 'answer_starts': [0]}),
    }
    return examples[mode]

class TestDureaderRobust(CpuCommonTest):
    """
    clue tnews case
    """
    def setUp(self):
        """
        check input params & datasets all flies
        """
        self.config['path_or_read_func'] = 'dureader_robust'
        self.config['splits'] = ['train','dev','test']

    def test_train_set(self):
        """
        check train.json length, label,text
        """
        expected_ds_num = 3
        expected_len = 14520
        expected_train= get_examples('train')
        ds = load_dataset(**self.config)
        self.check_output_equal(len(ds), expected_ds_num)
        self.check_output_equal(len(ds[0]), expected_len)

        self.check_output_equal(expected_train['question'], ds[0][0]['question'])
        self.check_output_equal(expected_train['id'], ds[0][0]['id'])
        self.check_output_equal(expected_train['title'], ds[0][0]['title'])
        self.check_output_equal(expected_train['context'], ds[0][0]['context'])
        self.check_output_equal(expected_train['answers'], ds[0][0]['answers'])
        self.check_output_equal(expected_train['answer_starts'], ds[0][0]['answer_starts'])


class TestDureaderRobustNoSplitDataFiles(CpuCommonTest):
    """
    check no splits 
    """
    def setUp(self):
        self.config['path_or_read_func'] = 'dureader_robust'

    @util.assert_raises
    def test_no_split_datafiles(self):
        load_dataset(**self.config)


if __name__ == "__main__":
    unittest.main()
