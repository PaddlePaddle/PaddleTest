# encoding: utf-8
"""
paddlecv 测试框架
"""


import subprocess
import sys
import os
import os.path
import shutil
import yaml
import paddle
import pytest
import allure
from utils.utils import exit_check_fucntion, allure_step


def prepare_repo():
    """
    prepare_repo
    """

    print("This is Repo Init!")
    os.system("git clone -b develop https://github.com/paddlepaddle/models.git --depth 1")
    os.chdir("models/paddlecv")
    os.system("python -m pip install .")
    # os.sysytem('python -m pip install -r requirements.txt')
    cmd = "python -m pip install -r requirements.txt"
    repo_result = subprocess.getstatusoutput(cmd)
    exit_code = repo_result[0]
    output = repo_result[1]
    print(output)
    assert exit_code == 0, "git clone models failed!   log information:%s" % (output)


class TestPaddleCVPredict:
    """
    TestPaddleCVPredict
    """

    def __init__(self, model=""):
        """
        __init__
        """
        self.model = model
        base_dir = os.path.split(os.path.realpath(__file__))[0]
        os.chdir(base_dir)
        self.model_config = yaml.load(open("utils/paddlecv.yml", "rb"), Loader=yaml.Loader)
        self.yml = self.model_config[self.model]["yml"]
        self.input = self.model_config[self.model]["input"]
        self.output_json = self.model_config[self.model]["output_json"]
        self.output_vis = self.model_config[self.model]["output_vis"]

    def test_cv_predict(self, run_mode="paddle", device="CPU"):
        """
        test_cv_predict
        """
        assert os.path.exists("models/paddlecv"), "models/paddlecv not exist!"
        os.chdir("models/paddlecv")
        if os.path.exists("output"):
            shutil.rmtree("output")
        cmd = "python -u tools/predict.py --config=%s --input=%s --run_mode=%s --device=%s" % (
            self.yml,
            self.input,
            run_mode,
            device,
        )
        print(cmd)
        result = subprocess.getstatusoutput(cmd)
        exit_code = result[0]
        output = result[1]
        allure_step(cmd)
        exit_check_fucntion(exit_code, output, self.output_vis, self.output_json, self.input)

    def test_wheel_predict(self):
        """
        test_wheel_predict
        """
        os.chdir("models/paddlecv")
        if os.path.exists("output"):
            shutil.rmtree("output")
        cmd = 'python -c "from paddlecv import PaddleCV; paddlecv = PaddleCV(task_name=%s); res = paddlecv(%s)"' % (
            self.model,
            self.input,
        )
        print(cmd)
        from paddlecv import PaddleCV

        try:
            paddlecv = PaddleCV(task_name=self.model)
            res = paddlecv(self.input)
        except Exception as e:
            print(repr(e))
            exit_check_fucntion(1, e, self.output_vis, self.output_json, self.input)

        else:
            # exit_check_fucntion(0, res, self.output_vis, self.output_json, self.input)
            print(res)
            pass
