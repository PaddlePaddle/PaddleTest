"""
grad run
"""

import os
import sys

curPath = os.path.abspath(os.path.dirname("utils"))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from utils.weaktrans import WeakTrans, Framework
from utils.yaml_loader import YamlLoader
import numpy as np
import paddle
import jax
from gradtest import JaxTest
from gradtrans import GradTrans


class RunCase(object):
    """
    run case class
    """

    def __init__(self, file_dir):
        """
        initialize
        """
        self.file_dir = file_dir
        self.yaml = YamlLoader(self.file_dir)

    def get_all_case_name(self):
        """
        get competitor case name
        """
        cases = self.yaml.get_all_case_name()
        # 返回有竞品测试的case_name
        case_list = []
        for case_name in cases:
            case = self.yaml.get_case_info(case_name)
            if case["info"].get("jax"):
                case_list.append(case_name)
        return case_list

    def get_docstring(self, case_name):
        """
        get docstring
        """
        case = self.yaml.get_case_info(case_name)
        return case["info"]["desc"]

    def run_case(self, case_name):
        """
        run case
        """
        case = self.yaml.get_case_info(case_name)
        self.exec(case)

    def exec(self, case):
        """
        actuator
        """
        trans = GradTrans(case)
        apis = trans.get_function()
        paddle_ins = trans.get_paddle_ins()
        jax_ins = trans.get_jax_ins()
        init_actor = trans.get_init_actor()

        jt = JaxTest(apis)
        jt.init_ad = init_actor
        jt.run(paddle_ins, jax_ins)


if __name__ == "__main__":
    obj = RunCase("../yaml/%s.yml" % sys.argv[1])
    r = obj.get_docstring("sin_0")
    print(r)
    obj.run_case("sin_0")
