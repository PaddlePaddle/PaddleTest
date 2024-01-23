# encoding: utf-8
"""
执行case前：生成yaml，设置特殊参数，改变监控指标
"""
import os
import logging
import platform

logger = logging.getLogger("ce")


class PaddleSeg_Case_Start(object):
    """
    自定义环境准备
    """

    def __init__(self):
        """
        初始化变量
        """
        self.reponame = os.getenv("reponame")
        self.mode = os.getenv("mode")
        self.case_step = os.getenv("case_step")
        self.case_name = os.getenv("case_name")
        self.qa_yaml_name = os.getenv("qa_yaml_name")

    def build_prepare(self):
        """
        执行准备过程
        """
        if str(os.getenv("SOT_EXPORT_FLAG")) == "True":
            os.environ["SOT_EXPORT"] = f"Layer_cases/{self.qa_yaml_name}_{self.case_name}_{self.case_step}"
            logger.info("set org SOT_EXPORT as {}".format(os.getenv("SOT_EXPORT")))
        # sysstr = platform.system()
        # if sysstr == "Linux":
        #     logger.info("###kill python process")
        #     pid = os.getpid()
        #     cmd = """ps aux| grep python | grep -v main.py |grep -v %s | awk '{print $2}'| xargs kill -9;""" % pid
        #     os.system(cmd)


def run():
    """
    执行入口
    """
    model = PaddleSeg_Case_Start()
    model.build_prepare()
    return 0


if __name__ == "__main__":
    # args = None
    # model = PaddleSeg_Case_Start(args)
    # model.build_prepare()
    run()
