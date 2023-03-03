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
        ({'text': '\nFrom flickr.com: Money {MID-161793} <p>Money ( <a href="https://farm8.static.flickr.com/7020/6551534889_9c8ae52997.jpg" type="external">Image</a> by <a href="https://www.flickr.com/people/68751915@N05/" type="external">401(K) 2013</a>) <a href="https://creativecommons.org/licenses/by-sa/2.0/" type="external">Permission</a> <a type="internal">Details</a> <a type="internal">DMCA</a></p> '
        'No Pill Can Stop Tinnitus, But This 1 Weird Trick Can <p>The walls are closing in on Congress.</p> <p>Terrifying walls of water from Hurricanes Harvey and Irma, which, when the damage is totaled, could rise to a half trillion dollars. The Walls of War: The multi-trillion dollar ongoing cost of Afghanistan, Iraq and other interventions. The crumbling walls of the U.S. infrastructure, which need at '
        'least $3 trillion to be repaired or replaced. A wall of 11 million undocumented immigrants, whose deportation could easily cost $200 billion. The planned wall at the Mexican border, which some estimates place at $67 billion. Then there is the Wall of All, the $20 trillion national debt. The walls of debt are closing in.'
        '</p> <p>At moments of crisis in our nation, in addition to invoking the assistance of Higher powers, we can call upon the Constitution for guidance.</p> <p>Article I, Section 8, of the U.S. Constitution contains a long-forgotten provision, &quot;the coinage clause,&quot; which empowered Congress &quot;to coin (create) Money.&quot; The ability to create money to meet the needs of the nation is a sovereign power, which enables a nation to have control of its own destiny.</p> <p>'
        'The same article indicates the Founders anticipated having to borrow money on the full faith and credit of the United States. Enter the Funding Act of 1790, which assumed and paid off the debt of the colonies and retired the financial obligations of the newly created states now united. This was a powerful, object lesson in debt retirement, relevant today.'
        '</p> <p>It is abundantly clear from a plain reading of the coinage clause that the Founders never intended that the only way the government was to be funded was to borrow money.</p> <p>The needs of the nation were to come from a system of not borrowing wherein money was a neutral value of exchange connecting resources, people and needs, without debt attached.</p> <p>'
        'In 1913, the passage of the Federal Reserve Act ceded the constitutional power to create money (and control of our national destiny), to the Federal Reserve, a quasi-private central bank. At this fateful point, the only way money could be brought into being was to borrow it, whereby money became equated with debt. The money system transited from public control to private control, and there it has remained.</p> <p>'
        'Instead of following the path set forth by the Founders to create money directly, our government became obliged to borrow from private banks, which assumed the sovereign power to create money from nothing and then loan it to the government, turning on its head the intention of the Founders.</p> <p>As a member of Congress, I came to the conclusion that while the debate over taxation was interesting, it was wholly insufficient. One must first study how money is created, before one can sensibly have a discussion of how it is to be taxed.</p> <p>'
        'With the help of staff, I spent a full five years working with legislative counsel to come up with a way to realign with the founding principles, to reclaim and to re-establish for our nation the sovereign power to create money.</p> <p>The vehicle was H.R. 2990, the National Emergency Employment Defense (NEED Act), which articulates why the current debate over the debt ceiling should lead directly to a debate about monetary policy, and the origins of the debt-based economic system.</p> '
        'How To Easily Kill All Indoor Odor, Mold, And Bacteria — Without Lifting A Finger No More Tinnitus (Ear Ringing) If You Do This Immediately <p>In our work on the NEED Act, we propound that the present monetary system has led to a concentration of wealth, expansion of national debt, excessive reliance on taxation, devaluation of the currency, increases in the cost of public infrastructure, unemployment and underemployment and the erosion of the ability of Congress to meet the needs of the American people.</p> <p>'
        'This system has been a source of financial instability where the banks\' ability to create money out of nothing has become a financial liability for the American taxpayers. When banks engaged in speculative lending, turning the financial system into a casino, they were bailed out while millions of Americans lost their homes. No surprise that today we are told there is not enough money for creating jobs, rebuilding America, health care, education and retirement security. But there is always money to bail out the banks.</p> <p>'
        'Let us take the opportunity afforded in the debate over the debt ceiling to regain control of our sovereignty and our national destiny. We can have a future of abundance instead of poverty, but we must first take down the wall which separates us from our true sovereignty, the power to coin and create money.</p> <p>Let us return to first principles, and reclaim the constitutional power to coin and create United States money and spend it into circulation to meet the needs of the nation and reduce taxes.</p> <p>'
        'Two hundred and thirty years ago this month, delegates from 13 states gathered in a constitutional convention, which set the stage for ratification. Let us summon that same revolutionary spirit and its wisdom to guide us in the days ahead.</p> Seniors Can\'t Get Enough of This Sweet Treat That Has Shown to Turn Back the Clock on Alzheimer\'s From flickr.com: Money {MID-161793} <p>Money ( <a href="https://farm8.static.flickr.com/7020/6551534889_9c8ae52997.jpg" type="external">Image</a> by <a href="https://www.flickr.com/people/68751915@N05/" type="external">401(K) 2013</a>) <a href="https://creativecommons.org/licenses/by-sa/2.0/" type="external">Permission</a> <a type="internal">Details</a> <a type="internal">DMCA</a></p> <p>The walls are closing in on Congress.</p> <p>'
        'Terrifying walls of water from Hurricanes Harvey and Irma, which, when the damage is totaled, could rise to a half trillion dollars. The Walls of War: The multi-trillion dollar ongoing cost of Afghanistan, Iraq and other interventions. The crumbling walls of the U.S. infrastructure, which need at least $3 trillion to be repaired or replaced. A wall of 11 million undocumented immigrants, whose deportation could easily cost $200 billion. The planned wall at the Mexican border, which some estimates place at $67 billion. Then there is the Wall of All, the $20 trillion national debt. The walls of debt are closing in.</p> <p>At moments of crisis in our nation, in addition to invoking the assistance of Higher powers, we can call upon the Constitution for guidance.</p> <p>'
        'Article I, Section 8, of the U.S. Constitution contains a long-forgotten provision, &quot;the coinage clause,&quot; which empowered Congress &quot;to coin (create) Money.&quot; The ability to create money to meet the needs of the nation is a sovereign power, which enables a nation to have control of its own destiny.</p> <p>The same article indicates the Founders anticipated having to borrow money on the full faith and credit of the United States. Enter the Funding Act of 1790, which assumed and paid off the debt of the colonies and retired the financial obligations of the newly created states now united. This was a powerful, object lesson in debt retirement, relevant today.</p> <p>'
        'It is abundantly clear from a plain reading of the coinage clause that the Founders never intended that the only way the government was to be funded was to borrow money.</p> <p>The needs of the nation were to come from a system of not borrowing wherein money was a neutral value of exchange connecting resources, people and needs, without debt attached.</p> <p>In 1913, the passage of the Federal Reserve Act ceded the constitutional power to create money (and control of our national destiny), to the Federal Reserve, a quasi-private central bank. At this fateful point, the only way money could be brought into being was to borrow it, whereby money became equated with debt. The money system transited from public control to private control, and there it has remained.</p> <p>'
        'Instead of following the path set forth by the Founders to create money directly, our government became obliged to borrow from private banks, which assumed the sovereign power to create money from nothing and then loan it to the government, turning on its head the intention of the Founders.</p> <p>As a member of Congress, I came to the conclusion that while the debate over taxation was interesting, it was wholly insufficient. One must first study how money is created, before one can sensibly have a discussion of how it is to be taxed.</p> <p>With the help of staff, I spent a full five years working with legislative counsel to come up with a way to realign with the founding principles, to reclaim and to re-establish for our nation the sovereign power to create money.</p> <p>'
        'The vehicle was H.R. 2990, the National Emergency Employment Defense (NEED Act), which articulates why the current debate over the debt ceiling should lead directly to a debate about monetary policy, and the origins of the debt-based economic system.</p> How To Easily Kill All Indoor Odor, Mold, And Bacteria — Without Lifting A Finger Trump to End the Dollar as We Know It by November 8, 2018?', 'label': 1}),
    }
    return examples[mode]

class TestHYP(CpuCommonTest):
    """
    clue tnews case
    """
    def setUp(self):
        """
        check input params & datasets all flies
        """
        self.config['path_or_read_func'] = 'hyp'
        self.config['splits'] = ['train', 'dev','test']

    def test_train_set(self):
        """
        check train.json length, label,text
        """
        expected_ds_num = 3
        expected_len = 516
        expected_train= get_examples('train')
        ds = load_dataset(**self.config)
        self.check_output_equal(len(ds), expected_ds_num)
        self.check_output_equal(len(ds[0]), expected_len)

        self.check_output_equal(expected_train['text'], ds[0][0]['text'])
        self.check_output_equal(int(expected_train['label']), ds[0][0]['label'])




class TestHYPNoSplitDataFiles(CpuCommonTest):
    """
    check no splits 
    """
    def setUp(self):
        self.config['path_or_read_func'] = 'hyp'

    @util.assert_raises
    def test_no_split_datafiles(self):
        load_dataset(**self.config)


if __name__ == "__main__":
    unittest.main()
