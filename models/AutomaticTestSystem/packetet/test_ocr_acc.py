import pytest
import numpy as np
import subprocess
import re
import sys
import yaml
import platform
import os.path

from ModelsTestFramework import RepoInit
from ModelsTestFramework import RepoDataset
from ModelsTestFramework import TestOcrModelFunction
from utility import * 

def get_model_list():
    import sys
    result = []
    category=[]
    with open('models_list_test.yaml') as f:
      lines = f.readlines()
      for line in lines:
         r = re.search('/(.*)/', line)
         result.append(line.strip('\n'))
         print(r.group(1))
         print(line)
         print("************************")
         category.append(r.group(1))
    return result

@pytest.fixture()
def login():
    print("登录")
    return 8

def setup_module():
    """
    """
    # RepoInit(repo='PaddleOCR')
    # RepoDataset(cmd='''cd PaddleOCR; ln -s /ssd1/panyan/data/train_data train_data;''') 
    RepoDataset()



@pytest.mark.parametrize('yml_name', get_model_list())
def test_rec_accuracy_get_pretrained_model(yml_name):
    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    r = re.search('/(.*)/', yml_name)
    category=r.group(1)
    print(category)
    model = TestOcrModelFunction(model=model_name, yml=yml_name, category=category)
    model.test_ocr_get_pretrained_model()

# @pytest.mark.skip
@pytest.mark.parametrize('yml_name', get_model_list())
@pytest.mark.parametrize("use_gpu", [True])
def test_rec_accuracy_eval(yml_name, use_gpu):
    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    r = re.search('/(.*)/', yml_name)
    category=r.group(1)
    print(category)
    model = TestOcrModelFunction(model=model_name, yml=yml_name, category=category)
    model.test_ocr_eval(use_gpu)

@pytest.mark.parametrize('yml_name', get_model_list())
@pytest.mark.parametrize("use_gpu", [True])
def test_rec_accuracy_infer(yml_name, use_gpu, login):
    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    r = re.search('/(.*)/', yml_name)
    category=r.group(1)
    print(category)
    print(login)
    model = TestOcrModelFunction(model=model_name, yml=yml_name, category=category)
    model.test_ocr_rec_infer(use_gpu)

@pytest.mark.parametrize('yml_name', get_model_list())
@pytest.mark.parametrize("use_gpu", [True,False])
def test_rec_accuracy_export_model(yml_name, use_gpu):
    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    r = re.search('/(.*)/', yml_name)
    category=r.group(1)
    print(category)
    model = TestOcrModelFunction(model=model_name, yml=yml_name, category=category)
    model.test_ocr_export_model(use_gpu)

@pytest.mark.parametrize('yml_name', get_model_list())
@pytest.mark.parametrize("enable_mkldnn", [True,False])
def test_rec_accuracy_predict_mkl(yml_name, enable_mkldnn):
    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    r = re.search('/(.*)/', yml_name)
    category=r.group(1)
    print(category)
    model = TestOcrModelFunction(model=model_name, yml=yml_name, category=category)
    model.test_ocr_rec_predict(False, 0, enable_mkldnn)

@pytest.mark.parametrize('yml_name', get_model_list())
@pytest.mark.parametrize("use_tensorrt", [True,False])
def test_rec_accuracy_predict_trt(yml_name, use_tensorrt):
    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    r = re.search('/(.*)/', yml_name)
    category=r.group(1)
    print(category)
    model = TestOcrModelFunction(model=model_name, yml=yml_name, category=category)
    model.test_ocr_rec_predict(True, use_tensorrt, 0)

@pytest.mark.skip
@pytest.mark.parametrize('yml_name', get_model_list())
@pytest.mark.parametrize("use_gpu", [True])
def test_rec_funtion_train(yml_name, use_gpu):
    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    r = re.search('/(.*)/', yml_name)
    category=r.group(1)
    print(category)
    model = TestOcrModelFunction(model=model_name, yml=yml_name, category=category)
    model.test_ocr_train(use_gpu)

@pytest.mark.skip
@pytest.mark.mac
@pytest.mark.parametrize('yml_name', get_model_list())
def test_rec_funtion_mac(yml_name):
    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    r = re.search('/(.*)/', yml_name)
    category=r.group(1)
    print(category)
    model = TestOcrModelFunction(model=model_name, yml=yml_name, category=category)
#    model.test_ocr_train(False)
#    model.test_ocr_get_pretrained_model()
    model.test_ocr_eval(False)
#    model.test_ocr_rec_infer(False)
#    model.test_ocr_export_model(False)
#    model.test_ocr_rec_predict(False, False, False)

# if __name__ == '__main__':

def main():
    case='test_ocr_acc.py::test_rec_accuracy_'.join(args.stage)
#    pytest.main([case,'-sv'])
    pytest.main(['test_ocr_acc.py::test_rec_accuracy_infer','-sv'])
