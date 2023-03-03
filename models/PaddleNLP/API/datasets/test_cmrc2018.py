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
        ({'id': 'TRAIN_186_QUERY_0', 'title': '范廷颂', 'context': '范廷颂枢机（，），圣名保禄·若瑟（），是越南罗马天主教枢机。1963年被任为主教；'
        '1990年被擢升为天主教河内总教区宗座署理；1994年被擢升为总主教，同年年底被擢升为枢机；2009年2月离世。范廷颂于1919年6月15日在越南宁平省天主教发艳教'
        '区出生；童年时接受良好教育后，被一位越南神父带到河内继续其学业。范廷颂于1940年在河内大修道院完成神学学业。范廷颂于1949年6月6日在河内的主教座堂晋铎；'
        '及后被派到圣女小德兰孤儿院服务。1950年代，范廷颂在河内堂区创建移民接待中心以收容到河内避战的难民。1954年，法越战争结束，越南民主共和国建都河内，当时'
        '很多天主教神职人员逃至越南的南方，但范廷颂仍然留在河内。翌年管理圣若望小修院；惟在1960年因捍卫修院的自由、自治及拒绝政府在修院设政治课的要求而被捕。'
        '1963年4月5日，教宗任命范廷颂为天主教北宁教区主教，同年8月15日就任；其牧铭为「我信天主的爱」。由于范廷颂被越南政府软禁差不多30年，因此他无法到所属堂'
        '区进行牧灵工作而专注研读等工作。范廷颂除了面对战争、贫困、被当局迫害天主教会等问题外，也秘密恢复修院、创建女修会团体等。1990年，教宗若望保禄二世在同年'
        '6月18日擢升范廷颂为天主教河内总教区宗座署理以填补该教区总主教的空缺。1994年3月23日，范廷颂被教宗若望保禄二世擢升为天主教河内总教区总主教并兼天主教谅山'
        '教区宗座署理；同年11月26日，若望保禄二世擢升范廷颂为枢机。范廷颂在1995年至2001年期间出任天主教越南主教团主席。2003年4月26日，教宗若望保禄二世任命天主'
        '教谅山教区兼天主教高平教区吴光杰主教为天主教河内总教区署理主教；及至2005年2月19日，范廷颂因获批辞去总主教职务而荣休；吴光杰同日真除天主教河内总教区总主教职务。'
        '范廷颂于2009年2月22日清晨在河内离世，享年89岁；其葬礼于同月26日上午在天主教河内总教区总主教座堂举行。',
        'question': '范廷颂是什么时候被任为主教的？', 'answers': ['1963年'], 'answer_starts': [30]}),
    }
    return examples[mode]

class TestCMRC2018(CpuCommonTest):
    """
    clue tnews case
    """
    def setUp(self):
        """
        check input params & datasets all flies
        """
        self.config['path_or_read_func'] = 'cmrc2018'
        self.config['splits'] = ['train','dev','trial']

    def test_train_set(self):
        """
        check train.json length, label,text
        """
        expected_ds_num = 3
        expected_len = 10142
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


class TestCMRC2018NoSplitDataFiles(CpuCommonTest):
    """
    check no splits 
    """
    def setUp(self):
        self.config['path_or_read_func'] = 'cmrc2018'

    @util.assert_raises
    def test_no_split_datafiles(self):
        load_dataset(**self.config)


if __name__ == "__main__":
    unittest.main()
