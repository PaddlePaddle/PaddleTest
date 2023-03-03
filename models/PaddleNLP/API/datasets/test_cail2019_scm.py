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
                "text_a": "原告李某1。委托代理人蒋水光，湘阴县南湖法律服务所法律工作者。被告夏某1。\n\n原告李某1诉称，2015年3月6日，被告夏某1因欠缺资金，向丰辉借款70000元。因丰辉又欠他70000元，"
                "2015年3月23日，他向丰辉追收欠款时，丰辉将被告夏某1所欠其70000元债权转让予他，被告夏某1同意转让并向他出具欠条一张。后被告夏某1经他多次催要，至今尚未归还本金及利息。为维护他的合法权益，"
                "特起诉至法院，请求法院依法判决：1、被告夏某1偿还其本金70000元及利息；2、由被告夏某1承担本案诉讼费。被告夏某1未提出答辩，亦未提交任何证据，本院视为其放弃答辩、举证、质证的权利，由此造"
                "成对其不利的法律后果由其自行承担。经审理查明，原告李某1与被告夏某1经人介绍相识。被告夏某1因资金周转困难，向丰辉借款70000元。丰辉因资金周转困难向原告李某1借款70000元。2015年3月23日，"
                "三方在原告李某1家里达成一致意见，由被告夏某1向原告李某1归还借款70000元，归还时间为2016年3月23日之前，同时被告夏某1向原告李某1出具欠条一张，内容为：“今欠到李某1人币柒万元整。（￥70000元）"
                "欠款归还之日李某1将丰辉打给我7万元收条一并归还。证明：凭此条兑换丰辉收条李某12015年3月23日夏某1归还时间一年之内430624195801035630”。后原告李某1多次催要未果，遂诉至法院。以上事实有原告"
                "当庭陈述、欠条及庭审笔录等在卷证实，足以认定。\n",
                "text_b": "原告：牛某1，男，1972年11月10日出生，汉族，无业，住山西省太原市。委托诉讼代理人：李晓星，山西新国泰律师事务所律师。委托诉讼"
                "代理人：崔飞杰，山西新国泰律师事务所律师。被告：山西智伟基业房地产开发有限公司，住山西省太原市小店区通达西街29号7-13号房，统一社会信用代码×××。法定代表人：李欣，总经理。被告：冯某1，男，"
                "1970年6月29日出生，汉族，住山西省太原市。被告：康某1，女，1973年7月26日出生，汉族，住山西省太原市。以上被告共同委托诉讼代理人：李建业，男，1955年8月30日出生，汉族，山西智伟基业房地产开"
                "发有限公司职工，住山西省太原市。\n\n原告牛某1向本院提出诉讼请求：1．请求法院判令三被告立即共同连带归还原告借款本金3000000元，并按照年利率24％的标准支付原告自2013年6月10日起至三被告实际"
                "还清全部欠款之日的利息，该利息暂计至2017年11月9日为3230000元；2．请求法院判令三被告承担本案全部诉讼费用。事实和理由：2011年11月2日，原告与被告冯某1、被告康某1签订了《借款协议书》，约定"
                "原告出借给被告冯某1、被告康某1人民币300万元，借款期限为12个月，自2011年11月2日至2012年10月31日。双方约定按每月3％计算利息，被告按季度向原告支付利息。上述合同签订后，原告依约向被告支付了"
                "全部款项，但被告一直未能按时支付利息，借款期限届满后也未能归还本金。2014年2月10日，被告山西智伟基业房地产开发有限公司向原告出具《承诺书》，明确了其向原告借款的事实，并承诺于2014年3月、6月"
                "向原告支付利息，于2014年11月2日前向原告还清全部本息。该承诺书出具后，原告与被告冯某1、被告康某1于2014年3月5日签订了《借款补充协议书》，约定将前述借款延期至2014年11月2日。但借款期限届满后"
                "三被告仍未依约还款，经原告多次催要无果，故诉至法院，请求法院依法支持原告的诉讼请求。被告山西智伟基业房地产开发有限公司、冯某1、康某1承认原告牛某1提出的全部诉讼请求。\n",
                "text_c": "原告："
                "王某1，女，1988年6月3日出生，汉族，无固定职业，住哈尔滨市道里区。被告：路某1，男，1987年9月9日出生，汉族，无固定职业，（户籍地）住哈尔滨市南岗区，现住哈尔滨市道里区。\n\n原告王某1向本院提"
                "出诉讼请求：1．判令路某1给付王某1借款本金7.7万元，利息从借贷日开始计算到实际给付之日止；2．由路某1承担本案诉讼费用。事实和理由：王某1与路某1经业务关系相识，路某1因经营需要于2017年1月24日向"
                "王某1借款5万元，约定月利息2％，2017年3月17日向王某1借款27000元。路某1承诺2017年5月1日前偿还两笔借款，还款期限届满后，王某1多次找到路某1追索借款未果，故诉至法院。被告路某1未出庭，未答辩。"
                "原告为证实其诉讼请求成立向本院提交了两份证据，1.2017年1月24日，路某1出具的借条一份；证明路某1向王某1第一次借款5万元的事实，利息约定月利率2％返给王某1，还款期限为借款之日起至2017年5月1日止。"
                "2．借条一份；证明被告2017年3月17日向路某1借款27000元，月利息2％，2017年5月1还清，这一条是后补的。根据当事人的陈述和经审查确认的证据，本院认定事实如下：2017年1月14日，路某1向王某1借款50000元，"
                "并出具借条一份，约定：借款日期为2017年1月24日至2017年5月1日；借款利息为月利息2％。后路某1又向王某1借款，2017年5月17日，路某1向王某1出具借条一份，约定：借款金额27000元，借款日期为2017年3月17日"
                "至2017年5月1日，借款利息为月利息2％。王某1多次催讨未果，诉至法院。\n",
                "label": 1,
            }
        ),
    }
    return examples[mode]


class TestCail2019_scm(CpuCommonTest):
    """
    clue tnews case
    """

    def setUp(self):
        """
        check input params & datasets all flies
        """
        self.config["path_or_read_func"] = "cail2019_scm"
        self.config["splits"] = ["train", "dev", "test"]

    def test_train_set(self):
        """
        check train.json length, label,text
        """
        expected_ds_num = 3
        expected_len = 5102
        expected_train = get_examples("train")
        ds = load_dataset(**self.config)
        self.check_output_equal(len(ds), expected_ds_num)
        self.check_output_equal(len(ds[0]), expected_len)

        self.check_output_equal(expected_train["text_a"], ds[0][0]["text_a"])
        self.check_output_equal(int(expected_train["label"]), ds[0][0]["label"])


class TestCail2019NoSplitDataFiles(CpuCommonTest):
    """
    check no splits
    """

    def setUp(self):
        self.config["path_or_read_func"] = "cail2019_scm"

    @util.assert_raises
    def test_no_split_datafiles(self):
        load_dataset(**self.config)


if __name__ == "__main__":
    unittest.main()
