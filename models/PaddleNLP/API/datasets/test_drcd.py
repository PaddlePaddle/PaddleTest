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
        ({'id': '1001-10-1', 'title': '广州', 'context': '2010年引進的廣州快速公交運輸系統，屬世界第二大快速公交系統，日常載客量可達100萬人次，'
        '高峰時期每小時單向客流高達26900人次，僅次於波哥大的快速交通系統，平均每10秒鐘就有一輛巴士，每輛巴士單向行駛350小時。包括橋樑在內的站台是世界最'
        '長的州快速公交運輸系統站台，長達260米。目前廣州市區的計程車和公共汽車主要使用液化石油氣作燃料，部分公共汽車更使用油電、氣電混合動力技術。2012年'
        '底開始投放液化天然氣燃料的公共汽車，2014年6月開始投放液化天然氣插電式混合動力公共汽車，以取代液化石油氣公共汽車。2007年1月16日，廣州市政府全面'
        '禁止在市區內駕駛摩托車。違反禁令的機動車將會予以沒收。廣州市交通局聲稱禁令的施行，使得交通擁擠問題和車禍大幅減少。廣州白雲國際機場位於白雲區與花都'
        '區交界，2004年8月5日正式投入運營，屬中國交通情況第二繁忙的機場。該機場取代了原先位於市中心的無法滿足日益增長航空需求的舊機場。目前機場有三條飛機'
        '跑道，成為國內第三個擁有三跑道的民航機場。比鄰近的香港國際機場第三跑道預計的2023年落成早8年。', 
        'question': '廣州的快速公交運輸系統每多久就會有一輛巴士？', 'answers': ['10秒鐘'], 'answer_starts': [84]}),
    }
    return examples[mode]

class TestDRCD(CpuCommonTest):
    """
    clue tnews case
    """
    def setUp(self):
        """
        check input params & datasets all flies
        """
        self.config['path_or_read_func'] = 'drcd'
        self.config['splits'] = ['train','dev','test']

    def test_train_set(self):
        """
        check train.json length, label,text
        """
        expected_ds_num = 3
        expected_len = 26936
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


class TestDRCDNoSplitDataFiles(CpuCommonTest):
    """
    check no splits 
    """
    def setUp(self):
        self.config['path_or_read_func'] = 'drcd'

    @util.assert_raises
    def test_no_split_datafiles(self):
        load_dataset(**self.config)


if __name__ == "__main__":
    unittest.main()
