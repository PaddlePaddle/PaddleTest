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

      def get_loss(self, log_path):
          """
          获取loss值
          """
          step = []
          loss = []
          num = 0
          fl = open(log_path, 'r').readlines()
          for row in fl:
              if 'epoch:' in row.strip():
                  member = row.strip.split(',')
                  for item in member:
                      if "loss" in item:
                          loss_item = item.strip.split(':')
                          loss.append = float(loss_item[-1])
                          step.append(num)
                          num += 1
          return step, loss

    
      def draw_curve(self):
          """
          绘制曲线
          """
          logger.info("***draw curve start")
          #step1, loss1 = self.get_loss(self.standard_log_path)
          #step2, loss2 = self.get_loss(self.prim_log_path)
          #file_name=''
          #plt.
    
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
  if __name__ == "__main__":
      run()
