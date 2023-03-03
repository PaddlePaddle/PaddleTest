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
        ({'goal': [['START', '托马斯 · 桑斯特', '陈思宇'], ['托马斯 · 桑斯特', '出生 日期', '1990 - 5 - 16'], ['陈思宇', '出生 日期', '1990 - 5 - 16']], 
        'knowledge': [['托马斯 · 桑斯特', '血型', 'A型'], ['托马斯 · 桑斯特', '标签', '口碑 很好'], ['托马斯 · 桑斯特', '获奖', '移动迷宫_提名 _ ( 2015 ； 第17届 ) _ 青少年选择奖 _ 青少年选择奖 - 最佳 电影 火花'], 
        ['托马斯 · 桑斯特', '性别', '男'], ['托马斯 · 桑斯特', '职业', '演员'], ['托马斯 · 桑斯特', '领域', '明星'], ['托马斯 · 桑斯特', '星座', '金牛座'], ['陈思宇', '星座', '金牛座'], ['陈思宇', '毕业 院校', '北京电影学院'],
         ['陈思宇', '体重', '65kg'], ['陈思宇', '性别', '男'], ['陈思宇', '职业', '演员'], ['陈思宇', '领域', '明星'], ['托马斯 · 桑斯特', '评论', '第一次 看到 这 孩子 是 在 《 真爱至上 》 ， 萌 翻 了 ， 现在 长大 了 气质 不错'], 
         ['托马斯 · 桑斯特', '主要成就', '2004年 金卫星奖 年轻 男演员 奖 提名'], ['托马斯 · 桑斯特', '代表作', '神秘博士第三季']], 
        'conversation': ['知道 外国 有 个 明星 长 得 很 萌 吗 ？', '这个 还 真 不知道 呢 ， 请问 是 谁 啊 ？', '是 托马斯 · 桑斯特 ， 颜值 太 高 了 。', '哦 ， 没 应 说过 呢 ， 你 能 给 大体 说说 么 ？', '给 你 大体 说说 ， 他 口碑 很好 的 ， 也 很 有 才华 ， 我们 国家 有 个 小 哥哥 跟 他 一样 都是 1990年5月16日 出生 的 。', '是 谁 啊 ？', '陈思宇 ， 金牛座 的 ， 毕业 于 北京电影学院 。', '有 时间 了解 一下 。']}),
    }
    return examples[mode]

class TestDuconv(CpuCommonTest):
    """
    clue tnews case
    """
    def setUp(self):
        """
        check input params & datasets all flies
        """
        self.config['path_or_read_func'] = 'duconv'
        self.config['splits'] = ['train', 'dev', 'test_1', 'test_2']

    def test_train_set(self):
        """
        check train.json length, label,text
        """
        expected_ds_num = 4
        expected_len = 19858
        expected_train= get_examples('train')
        ds = load_dataset(**self.config)
        self.check_output_equal(len(ds), expected_ds_num)
        self.check_output_equal(len(ds[0]), expected_len)

        self.check_output_equal(expected_train['goal'], ds[0][0]['goal'])
        self.check_output_equal(expected_train['knowledge'], ds[0][0]['knowledge'])
        self.check_output_equal(expected_train['conversation'], ds[0][0]['conversation'])


class TestDuconvNoSplitDataFiles(CpuCommonTest):
    """
    check no splits 
    """
    def setUp(self):
        self.config['path_or_read_func'] = 'duconv'

    @util.assert_raises
    def test_no_split_datafiles(self):
        load_dataset(**self.config)


if __name__ == "__main__":
    unittest.main()
