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
        ({"label": 8, "label_des": "公共交通", "sentence": "闪客蜂是一款地铁快速购票软件。它能满足手机扫码乘地铁、乘公交，手机购地铁票、有轨电车单程票，手机购买蜂格咖啡、周边美食商品等服务，各大城市用户享受不同的业务功能。极速购票不带零钱手机购票三秒扫码进站，"
        "上下班快速通行节省时间；旅行出差线路导航旅行出差，查看地铁线路、站点详情，一目了然；发现周边爆品抢购地铁周边食、玩，不容错过，一键下单享优惠支持城市线上购买地铁单程票业务开通城市广州、青岛、南宁、西安、郑州、长沙、天津、武汉、南昌刷地铁、刷公交业务已开通城市长"
        "沙扫描二维码乘地铁业务开通城市广州购买有轨电车电子单程票业务开通城市珠海蜂格咖啡业务开通城市广州、北京线下购票业务开通城市哈尔滨联系我们客服电话4006408811微信服务号shnkephone微信订阅号蜂来疯趣新浪微博闪客蜂官博闪客蜂交流QQ群491307302闪客蜂南昌用户交流群"
        "463102052闪客蜂广州用户交流群776968417闪客蜂青岛用户交流群894602175闪客蜂南宁用户交流群894604440闪客蜂长沙用户交流群731658963闪客蜂哈尔滨用户交流群627810460更新内容Hi~小蜜蜂们，瞧一瞧4.3.5新版本了～1、订单订单合并，优化用户体验细节；2、发现在发现"
        "页增加地铁周边商品；3、优化修复已知bug；", "id": 536})
    }
    return examples[mode]

class TestFewClueIFLYTEK(CpuCommonTest):
    """
    clue tnews case
    """
    def setUp(self):
        """
        check input params & datasets all flies
        """
        self.config['path_or_read_func'] = 'fewclue'
        self.config['name'] = 'iflytek'
        self.config['splits'] = ['train_0', 'train_1', 'train_2', 'train_3', 'train_4', 'train_few_all', 'dev_0', 'dev_1', 'dev_2', 'dev_3', 'dev_4', 'dev_few_all', 'unlabeled', 'test', 'test_public']

    def test_train_set(self):
        """
        check train.json length, label,label_des, sentence,id
        """
        expected_ds_num = 15
        expected_len = 928
        expected_train= get_examples('train')
        ds = load_dataset(**self.config)
        self.check_output_equal(len(ds), expected_ds_num)
        self.check_output_equal(len(ds[0]), expected_len)

        self.check_output_equal(int(expected_train['label']), ds[0][0]['label'])
        self.check_output_equal(expected_train['label_des'], ds[0][0]['label_des'])
        self.check_output_equal(expected_train['sentence'], ds[0][0]['sentence'])
        self.check_output_equal(expected_train['id'], ds[0][0]['id'])


class TestFewclueIFLYTEKNoSplitDataFiles(CpuCommonTest):
    """
    check no splits 
    """
    def setUp(self):
        self.config['path_or_read_func'] = 'fewclue'
        self.config['task_name'] = 'iflytek'

    @util.assert_raises
    def test_no_split_datafiles(self):
        load_dataset(**self.config)


if __name__ == "__main__":
    unittest.main()
