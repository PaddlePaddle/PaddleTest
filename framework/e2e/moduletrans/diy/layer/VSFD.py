#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
VSFD
"""
import paddle
import paddle.nn.functional as F


class VSFD(paddle.nn.Layer):
    """VSFD"""

    def __init__(self, in_channels=512, pvam_ch=512, char_num=38):
        """init"""
        super(VSFD, self).__init__()
        self.char_num = char_num
        self.fc0 = paddle.nn.Linear(in_features=in_channels * 2, out_features=pvam_ch)
        self.fc1 = paddle.nn.Linear(in_features=pvam_ch, out_features=self.char_num)

    def forward(self, pvam_feature, gsrm_feature):
        """forward"""
        b, t, c1 = pvam_feature.shape
        b, t, c2 = gsrm_feature.shape
        combine_feature_ = paddle.concat([pvam_feature, gsrm_feature], axis=2)
        img_comb_feature_ = paddle.reshape(combine_feature_, shape=[-1, c1 + c2])
        img_comb_feature_map = self.fc0(img_comb_feature_)
        img_comb_feature_map = F.sigmoid(img_comb_feature_map)
        img_comb_feature_map = paddle.reshape(img_comb_feature_map, shape=[-1, t, c1])
        combine_feature = img_comb_feature_map * pvam_feature + (1.0 - img_comb_feature_map) * gsrm_feature
        img_comb_feature = paddle.reshape(combine_feature, shape=[-1, c1])

        out = self.fc1(img_comb_feature)
        return out
