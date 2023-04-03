# encoding: utf-8
"""
执行case后执行的文件
"""
import os
import sys
import json
import glob
import shutil
import argparse
import logging
import yaml
import wget
import numpy as np

logger = logging.getLogger("ce")


class PaddleDetection_End(object):
    """
    case执行结束后
    """
    
    def __init__(self):
        """
        初始化
        """
        self.reponame = os.environ["reponame"]
    
    def draw_curve(self):
        """
        绘制曲线
        """
        print("ok")
    
    def build_end(self):
        """
        执行准备过程
        """
        ret = 0
        ret = self.draw_curve()
        if ret:
            logger.info("draw curve failed!")
            return ret
        logger.info("draw curve end!")
        return ret


  def run():
      """
      执行入口
      """
      model = PaddleDetection_End()
      model.build_end()
      return 0
