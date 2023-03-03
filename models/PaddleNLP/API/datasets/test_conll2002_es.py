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
        ({'tokens': ['Melbourne', '(', 'Australia', ')', ',', '25', 'may', '(', 'EFE', ')', '.'], 
        'ner_tags': ['B-LOC', 'O', 'B-LOC', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'O', 'O'], 
        'pos_tags': ['NP', 'Fpa', 'NP', 'Fpt', 'Fc', 'Z', 'NC', 'Fpa', 'NC', 'Fpt', 'Fp']}),
    }
    return examples[mode]

class TestCONLL2022_ES(CpuCommonTest):
    """
    clue tnews case
    """
    def setUp(self):
        """
        check input params & datasets all flies
        """
        self.config['path_or_read_func'] = 'conll2002'
        self.config['name'] = 'es'
        self.config['splits'] = ['train', 'dev','test']

    def test_train_set(self):
        """
        check train.json length, label,label_desc, sentence, keywords
        """
        expected_ds_num = 3
        expected_len = 8324
        expected_train= get_examples('train')
        ds = load_dataset(**self.config)
        self.check_output_equal(len(ds), expected_ds_num)
        self.check_output_equal(len(ds[0]), expected_len)

        self.check_output_equal(expected_train['tokens'], ds[0][0]['tokens'])
        self.check_output_equal(expected_train['ner_tags'], ds[0][0]['ner_tags'])
        self.check_output_equal(expected_train['pos_tags'], ds[0][0]['pos_tags'])


class TestCONLL2022_ESNoSplitDataFiles(CpuCommonTest):
    """
    check no splits 
    """
    def setUp(self):
        self.config['path_or_read_func'] = 'conll2002'
        self.config['task_name'] = 'es'

    @util.assert_raises
    def test_no_split_datafiles(self):
        load_dataset(**self.config)


if __name__ == "__main__":
    unittest.main()
