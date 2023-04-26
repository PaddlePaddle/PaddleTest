#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
layer builder
"""

import paddle
import diy

# import ppdet
# from ppdet.modeling.proposal_generator.target_layer import RBoxAssigner

# import ppcls


class BuildLayer(object):
    """BuildLayer"""

    def __init__(self, layer_name, layer_param):
        """init"""
        # self.repo = repo
        self.layer_name = layer_name
        self.layer_param = layer_param
        if self.layer_param is not None:
            # 带有**标记的字符串转换为python object
            for k, v in self.layer_param.items():
                if isinstance(v, str):
                    if "**" in v:
                        try:
                            tmp = v
                            tmp = tmp.replace("**", "")
                            self.layer_param[k] = eval(tmp)
                        except:
                            self.layer_param[k] = v

    def get_layer(self):
        """get_layer"""
        if self.layer_param is not None:
            layer = eval(self.layer_name)(**self.layer_param)
        else:
            layer = eval(self.layer_name)()
        return layer
