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


def get_examples(mode="train"):
    """
    dataset[0][0] examples
    """
    examples = {
        "train": (
            {
                "id": 0,
                "abst": "无线传感器网络中实现隐私保护通用近似查询是具有挑战性的问题．文中提出一种无线传感器网络中隐私保护通用近似查询协议PGAQ．PGAQ将传感器节点编号和其采集数据隐藏于设计的数据结构中，"
                "在基站构造线性方程组解出直方图，根据直方图具有的统计信息，不泄露隐私地完成Top-k查询、范围查询、SUM、MAX/MIN、Median、Histogram等近似查询．PGAQ使用网内求和聚集以减少能量消耗，并且能够通过调节"
                "直方图划分粒度来平衡查询精度与能量消耗．PGAQ协议分为H-PGAQ和F-PGAQ两种模式．H-PGAQ模式使用数据扰动技术加强数据安全性，F-PGAQ使用过滤器减少连续查询通信量．通过理论分析和使用真实数据集实验验证了"
                "PGAQ的安全性和有效性．",
                "keyword": ["无线传感器网络", "数据聚集", "物联网", "近似查询"],
                "label": "1",
            }
        ),
        "dev": (
            {
                "id": 1,
                "abst": "为解决传统均匀FFT波束形成算法引起的3维声呐成像分辨率降低的问题,该文提出分区域FFT波束形成算法.远场条件下,以保证成像分辨率为约束条件,以划分数量最少为目标,采用遗传算法"
                "作为优化手段将成像区域划分为多个区域.在每个区域内选取一个波束方向,获得每一个接收阵元收到该方向回波时的解调输出,以此为原始数据在该区域内进行传统均匀FFT波束形成.对FFT计算过程进行优化,降低新算法"
                "的计算量,使其满足3维成像声呐实时性的要求.仿真与实验结果表明,采用分区域FFT波束形成算法的成像分辨率较传统均匀FFT波束形成算法有显著提高,且满足实时性要求.",
                "keyword": ["水声学", "FFT", "波束形成", "3维成像声呐"],
                "label": "1",
            }
        ),
        "test": (
            {
                "id": 2415,
                "abst": "总结了在对台风风神数值预报失效，预报路径偏东的情况下，预报人员抓住天气形势的细微变化对路径及时修正，并紧密结合用户需求，以“全程跟进，及时沟通，急用户之所急”的高度责任心和服务态度，使海上石油平台1500人安全撤离，"
                "避免了人员伤亡和重大经济损失的服务过程。经研究发现：(1)“风神”偏西侧的强对流发展、地面负变压中心、中层正涡度中心、高层正散度中心的存在，以及云图北侧带状黑体区的形成均有利于“风神”西折；(2)在500hPa图上，“风神”前期西南部有一低压环流中心，后期"
                "东北侧高压坝形成，造成“风神”两次北翘。",
                "keyword": ["正散度", "风神"],
            }
        ),
    }
    return examples[mode]


class TestFewClueCSL(CpuCommonTest):
    """
    clue tnews case
    """

    def setUp(self):
        """
        check input params & datasets all flies
        """
        self.config["path_or_read_func"] = "fewclue"
        self.config["name"] = "csl"
        self.config["splits"] = [
            "train_0",
            "train_1",
            "train_2",
            "train_3",
            "train_4",
            "train_few_all",
            "dev_0",
            "dev_1",
            "dev_2",
            "dev_3",
            "dev_4",
            "dev_few_all",
            "unlabeled",
            "test",
            "test_public",
        ]

    def test_train_set(self):
        """
        check train.json length, label,label_desc, sentence, keywords
        """
        expected_ds_num = 15
        expected_len = 32
        expected_train = get_examples("train")
        ds = load_dataset(**self.config)
        self.check_output_equal(len(ds), expected_ds_num)
        self.check_output_equal(len(ds[0]), expected_len)

        self.check_output_equal(expected_train["id"], ds[0][0]["id"])
        self.check_output_equal(expected_train["abst"], ds[0][0]["abst"])
        self.check_output_equal(expected_train["keyword"], ds[0][0]["keyword"])
        self.check_output_equal(expected_train["label"], ds[0][0]["label"])


class TestFewClueCSLNoSplitDataFiles(CpuCommonTest):
    """
    check no splits
    """

    def setUp(self):
        self.config["path_or_read_func"] = "fewclue"
        self.config["task_name"] = "csl"

    @util.assert_raises
    def test_no_split_datafiles(self):
        load_dataset(**self.config)


if __name__ == "__main__":
    unittest.main()
