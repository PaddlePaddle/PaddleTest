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
        ({"label": "11", "label_des": "薅羊毛", "sentence": "活动新用户领5元现金即日起成功注册酷划的新用户，填写邀请码80008即可领取5元现金奖励哦。小伙伴们不要错过，赶快来酷划锁屏领红包吧~每天解锁50次，一年可赚上千元酷划锁屏是一款可以自动赚钱的手机锁屏APP，"
        "每次解锁手机均有数额不等的现金奖励。赚到的钱可以极速提现到微信钱包、支付宝，也可以用来购买爆款商品。除手机解锁可以赚钱外，酷划还提供邀请赚、分享赚、高额赚等十余种赚钱方式，零成本可得高额回报。三年时间里，酷划已累计服务了超一亿名用户。在手赚软件中处于无可争议"
        "的领先地位。手机赚钱，用酷划锁屏就对了功能介绍手机解锁领现金每次解锁均有现金奖励锁屏精选新鲜资讯，边看边赚零花钱十种以上赚钱方式邀请赚、分享赚、试玩赚…多种赚钱方式任你选每天花上五分钟，酷划收入翻倍领邀请徒弟收入加倍邀请好友体验酷划，建立师徒关系徒弟挣钱，师傅"
        "得现金奖励提现购物任你挑选您的收入可提现，购物，话费充值微信一元提现，极速到账官网www.coohua.com微信公众号酷划在线客服邮箱help@coohua.com,每天解锁50次，一年可赚上千块,1、客户端个人中心电商相关功能下线2、锁屏改造客户端SDK"}),
        'dev':
        ({"label": "110", "label_des": "社区超市", "sentence": "朴朴快送超市创立于2016年，专注于打造移动端30分钟即时配送一站式购物平台，商品品类包含水果、蔬菜、肉禽蛋奶、海鲜水产、粮油调味、酒水饮料、休闲食品、日用品、外卖等。朴朴公司希望能以全新的商业模式，"
        "更高效快捷的仓储配送模式，致力于成为更快、更好、更多、更省的在线零售平台，带给消费者更好的消费体验，同时推动中国食品安全进程，成为一家让社会尊敬的互联网公司。,朴朴一下，又好又快,1.配送时间提示更加清晰友好2.保障用户隐私的一些优化3.其他提高使用体验的调整4.修复了一些已知bug"}),
        'test':
        ({"id": 0, "sentence": "大疆商城DJIStore，手机下单，快人一步，获得专享优惠，查找航拍点，了解热门的无人机资讯。优惠活动随时参与获得活动消息，专享移动端购机优惠。新品动态即时购买一站购物，查找门店，电池预约，以旧换新。探索附近航拍地点发现和分享航拍点，上帝"
        "视角，就在身旁。浏览热门的航拍资讯航拍佳作，进阶教程，新鲜玩法，应有尽有。大疆创新www.dji.com，全球知名无人机飞行器研发和生产商，占据全球超过半数的市场份额，客户遍布100多个国家，重新定义中国制造的魅力内涵。基于出色的技术，大疆成为全球科技领域前沿代表企业。"
        "联系我们请访问www.dji.com/contct更新内容新增「我的设备」，您可以更加快捷地购买到设备相关的配件，并轻松获得售后支持"}),
        'labels':
        ({"label": "0", "label_des": "打车"}),
    }
    return examples[mode]

class TestClueIFLYTEK(CpuCommonTest):
    """
    clue tnews case
    """
    def setUp(self):
        """
        check input params & datasets all flies
        """
        self.config['path_or_read_func'] = 'clue'
        self.config['name'] = 'iflytek'
        self.config['splits'] = ['train', 'dev','test','labels']

    def test_train_set(self):
        """
        check train.json length, label,label_des, sentence
        """
        expected_ds_num = 4
        expected_len = 12133
        expected_train= get_examples('train')
        ds = load_dataset(**self.config)
        self.check_output_equal(len(ds), expected_ds_num)
        self.check_output_equal(len(ds[0]), expected_len)

        self.check_output_equal(int(expected_train['label']), ds[0][0]['label'])
        self.check_output_equal(expected_train['label_des'], ds[0][0]['label_des'])
        self.check_output_equal(expected_train['sentence'], ds[0][0]['sentence'])


class TestClueIFLYTEKNoSplitDataFiles(CpuCommonTest):
    """
    check no splits 
    """
    def setUp(self):
        self.config['path_or_read_func'] = 'clue'
        self.config['task_name'] = 'iflytek'

    @util.assert_raises
    def test_no_split_datafiles(self):
        load_dataset(**self.config)


if __name__ == "__main__":
    unittest.main()
