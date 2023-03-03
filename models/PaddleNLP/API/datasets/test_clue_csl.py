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
        ({"id": 1, "abst": "目的探讨常见氧化铁纳米粒子几种神经干细胞标记技术的标记效率.材料与方法使用超顺磁性氧化铁纳米粒子(SPIO)和超微超顺磁性氧化铁纳米粒子(USPIO)以25μgFe/ml分别单独标记、"
        "与多聚赖氨酸(PLL)及脂质体联合标记神经干细胞,以未标记细胞做对照,采用普鲁士蓝染色评价细胞标记率,并采用4.7TMRIT2WI多回波序列测量T2弛豫率(R2)评价细胞内的铁摄取量,比较各组R2的差异.结果①普"
        "鲁士蓝染色结果:SPIO及USPIO单独标记组标记率为60％～70％,低于联合标记组的100％;②MRI结果:未标记细胞R2为(2.10±0.11)/s,SPIO、USPIO单独标记组细胞R2分别为(3.39±0.21)/s、(3.16±0.32)/s,"
        "SPIO-脂质体联合标记组及USPIO-脂质体联合标记组R2分别为(4.03±025)/s、(3.61±0.32)/s,SPIO-PLL联合标记组及USPIO-PLL联合标记组R2分别为(5.38±0.52)/s、(4.44±0.35)/s,SPIO、USPIO与"
        "PLL联合标记组R2大于SPIO、USPIO与脂质体联合标记组(P＜0.05);而与脂质体联合标记组R2大于单独标记组(P＜0.05);SPIO与USPIO单独标记细胞时R2差异无统计学意义(P＞0.05),SPIO与脂质体或PLL"
        "联合标记时R2高于USPIO(P＜0.05).结论SPIO、USPIO单独标记及与PLL、脂质体联合标记均可以成功标记神经干细胞,提高R2,其中SPIO与PLL联合标记效率最高.", "keyword": ["粒子", "铁化合物", "联合", "0.05"], "label": "0"}),
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

class TestClueCSL(CpuCommonTest):
    """
    clue tnews case
    """
    def setUp(self):
        """
        check input params & datasets all flies
        """
        self.config['path_or_read_func'] = 'clue'
        self.config['name'] = 'csl'
        self.config['splits'] = ['train', 'dev','test']

    def test_train_set(self):
        """
        check train.json length, label,label_desc, sentence, keywords
        """
        expected_ds_num = 3
        expected_len = 20000
        expected_train= get_examples('train')
        ds = load_dataset(**self.config)
        self.check_output_equal(len(ds), expected_ds_num)
        self.check_output_equal(len(ds[0]), expected_len)


        self.check_output_equal(expected_train['id'], ds[0][0]['id'])
        self.check_output_equal(expected_train['abst'], ds[0][0]['abst'])
        self.check_output_equal(expected_train['keyword'], ds[0][0]['keyword'])
        self.check_output_equal(int(expected_train['label']), ds[0][0]['label'])


class TestClueCSLNoSplitDataFiles(CpuCommonTest):
    """
    check no splits 
    """
    def setUp(self):
        self.config['path_or_read_func'] = 'clue'
        self.config['task_name'] = 'csl'

    @util.assert_raises
    def test_no_split_datafiles(self):
        load_dataset(**self.config)


if __name__ == "__main__":
    unittest.main()
