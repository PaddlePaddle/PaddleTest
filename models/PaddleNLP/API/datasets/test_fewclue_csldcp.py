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
        ({"content": "为保证发动机在不同的转速时都具有最佳点火提前角,同时提高摩托车的防盗能力,提出一种基于转速匹配的点火提前角和防盗控制方法.利用磁电机脉冲计算发动机转速,线生调整点火信号的触发延迟时间,"
        "实现点火提前角的精确控制.根据转速信息,结合GSM和GPS对点火器进行远程点火使能控制,设计数字点火器实现对摩托车进行跟踪与定位,并给出点火器的软硬件结构和详细设计.台架测试和道路测试表明所设计的基于发动机"
        "转速的数字点火器点火提前角控制精确,点火性能好,防盗能力强、范围广.", "label": "控制科学与工程", "id": 805}),
        'dev':
        ({"id": 1, "abst": "为解决传统均匀FFT波束形成算法引起的3维声呐成像分辨率降低的问题,该文提出分区域FFT波束形成算法.远场条件下,以保证成像分辨率为约束条件,以划分数量最少为目标,采用遗传算法"
        "作为优化手段将成像区域划分为多个区域.在每个区域内选取一个波束方向,获得每一个接收阵元收到该方向回波时的解调输出,以此为原始数据在该区域内进行传统均匀FFT波束形成.对FFT计算过程进行优化,降低新算法"
        "的计算量,使其满足3维成像声呐实时性的要求.仿真与实验结果表明,采用分区域FFT波束形成算法的成像分辨率较传统均匀FFT波束形成算法有显著提高,且满足实时性要求.", "keyword": ["水声学", "FFT", "波束形成", "3维成像声呐"], "label": "1"}),
        'test':
        ({"id": 2415, "abst": "总结了在对台风风神数值预报失效，预报路径偏东的情况下，预报人员抓住天气形势的细微变化对路径及时修正，并紧密结合用户需求，以“全程跟进，及时沟通，急用户之所急”的高度责任心和服务态度，使海上石油平台1500人安全撤离，"
        "避免了人员伤亡和重大经济损失的服务过程。经研究发现：(1)“风神”偏西侧的强对流发展、地面负变压中心、中层正涡度中心、高层正散度中心的存在，以及云图北侧带状黑体区的形成均有利于“风神”西折；(2)在500hPa图上，“风神”前期西南部有一低压环流中心，后期"
        "东北侧高压坝形成，造成“风神”两次北翘。", "keyword": ["正散度", "风神"]}),
    }
    return examples[mode]

class TestFewClueCSLDCP(CpuCommonTest):
    """
    clue tnews case
    """
    def setUp(self):
        """
        check input params & datasets all flies
        """
        self.config['path_or_read_func'] = 'fewclue'
        self.config['name'] = 'csldcp'
        self.config['splits'] = ['train_0', 'train_1', 'train_2', 'train_3', 'train_4', 'train_few_all', 'dev_0', 'dev_1', 'dev_2', 'dev_3', 'dev_4', 'dev_few_all', 'unlabeled', 'test', 'test_public']

    def test_train_set(self):
        """
        check train.json length, label, content, id
        """
        expected_ds_num = 15
        expected_len = 536
        expected_train= get_examples('train')
        ds = load_dataset(**self.config)
        self.check_output_equal(len(ds), expected_ds_num)
        self.check_output_equal(len(ds[0]), expected_len)


        self.check_output_equal(expected_train['id'], ds[0][0]['id'])
        self.check_output_equal(expected_train['content'], ds[0][0]['content'])
        self.check_output_equal(expected_train['label'], ds[0][0]['label'])


class TestFewClueCSLDCPNoSplitDataFiles(CpuCommonTest):
    """
    check no splits 
    """
    def setUp(self):
        self.config['path_or_read_func'] = 'fewclue'
        self.config['task_name'] = 'csldcp'

    @util.assert_raises
    def test_no_split_datafiles(self):
        load_dataset(**self.config)


if __name__ == "__main__":
    unittest.main()
