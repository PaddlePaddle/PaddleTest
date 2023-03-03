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
        ({'text': 'phone#design_features', 'text_pair': '今天有幸拿到了港版白色iPhone 5真机，试玩了一下，说说感受吧：'
        "1. 真机尺寸宽度与4/4s保持一致没有变化，长度多了大概一厘米，也就是之前所说的多了一排的图标。2. 真机重量比上一代轻了很多，"
        "个人感觉跟i9100的重量差不多。（用惯上一代的朋友可能需要一段时间适应了）3. 由于目前还没有版的SIM卡，无法插卡使用，有购买的朋友要注意了，"
        "并非简单的剪卡就可以用，而是需要去运营商更换新一代的SIM卡。4. 屏幕显示效果确实比上一代有进步，不论是从清晰度还是不同角度的视角，iPhone 5绝对要更上一层，"
        "我想这也许是相对上一代最有意义的升级了。5. 新的数据接口更小，比上一代更好用更方便，使用的过程会有这样的体会。6. 从简单的几个操作来讲速度比4s要快，这个不用"
        '测试软件也能感受出来，比如程序的调用以及照片的拍摄和浏览。不过，目前水货市场上坑爹的价格，最好大家可以再观望一下，不要急着出手。', 'label': 1}),
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
        self.config['path_or_read_func'] = 'seabsa16'
        self.config['name'] = 'phns'
        self.config['splits'] = ['train','test']

    def test_train_set(self):
        """
        check train.json length, label,sentence
        """
        expected_ds_num = 2
        expected_len = 1336
        expected_train= get_examples('train')
        ds = load_dataset(**self.config)
        self.check_output_equal(len(ds), expected_ds_num)
        self.check_output_equal(len(ds[0]), expected_len)

        self.check_output_equal(expected_train['text'], ds[0][0]['text'])
        self.check_output_equal(expected_train['text_pair'], ds[0][0]['text_pair'])
        self.check_output_equal(expected_train['label'], ds[0][0]['label'])


class TestSeasa16PhnsNoSplitDataFiles(CpuCommonTest):
    """
    check no splits 
    """
    def setUp(self):
        self.config['path_or_read_func'] = 'seabsa16'
        self.config['task_name'] = 'phns'

    @util.assert_raises
    def test_no_split_datafiles(self):
        load_dataset(**self.config)


if __name__ == "__main__":
    unittest.main()
