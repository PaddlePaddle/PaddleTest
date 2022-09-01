import pytest
import numpy as np
import subprocess
import re
import allure
import os

from RocmTestFramework import RepoInit
from RocmTestFramework import RepoRemove

def allure_step(cmd, output):
    with allure.step("运行指令：{}".format(cmd)):
           pass

def custom_instruction(cmd, model):
         repo_result=subprocess.getstatusoutput(cmd)
         exit_code=repo_result[0]
         output=repo_result[1]
         allure_step(cmd, output)
         allure.attach(output, model+'.log', allure.attachment_type.TEXT)
         assert exit_code == 0, " %s  failed!   log information:%s" % (model, output)

def set_case_list(dir_path=''):
    cmd='cd %s; find . -maxdepth 1 -name "test_*.py" | sort ' % (dir_path)
    print(cmd)
    repo_result=subprocess.getstatusoutput(cmd)
    exit_code=repo_result[0]
    output=repo_result[1]
    result=output
    print(result[0])
    return result

def get_case_list(filename='models_list.yaml'):
    import sys
    result = []
    with open(filename) as f:
      lines = f.readlines()
      for line in lines:
         result.append(line.strip('\n'))
    return result

def setup_module():
    """
    """
    RepoInit(repo='PaddleScience')
    cmd = '''cd PaddleScience; export PYTHONPATH=$PWD:$PYTHONPATH'''
    os.system(cmd)
    


def teardown_module():
    """
    """
    RepoRemove(repo='PaddleScience')


@allure.story('API')
@pytest.mark.parametrize('case_name', get_case_list('science_api.txt'))
def test_science_api(case_name):
    cmd='cd PaddleScience; export PYTHONPATH=$PWD:$PYTHONPATH; cd tests/test_api; python -m pytest -sv %s' % (case_name)
    custom_instruction(cmd, case_name)

@allure.story('modles')
@pytest.mark.skip(reason="release not supported")
@pytest.mark.parametrize('case_name', get_case_list('science_models.txt'))
def test_science_models(case_name):
    cmd='cd PaddleScience; export PYTHONPATH=$PWD:$PYTHONPATH; cd tests/test_models; python -m pytest -sv %s' % (case_name)
    custom_instruction(cmd, case_name)
