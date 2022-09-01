import pytest
import numpy as np
import subprocess
import re
import sys
import yaml
import platform
import os.path
import allure

from ModelsTestFramework import RepoInit
from ModelsTestFramework import RepoDataset
from ModelsTestFramework import TestOcrModelFunction


def get_model_list():
    import sys
    result = []
    category=[]
    with open('models_list_test_rec.yaml') as f:
      lines = f.readlines()
      for line in lines:
         r = re.search('/(.*)/', line)
         result.append(line.strip('\n'))
         print(r.group(1))
         print(line)
         print("************************")
         category.append(r.group(1))
    return result


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

@allure.story('infer')
@pytest.mark.parametrize('yml_name', get_model_list())
@pytest.mark.parametrize("use_gpu", [True,False])
def test_rec_accuracy_infer(yml_name, use_gpu):
    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    if use_gpu==True:
       hardware='_GPU'
    else:
       hardware='_CPU'
    allure.dynamic.title(model_name+hardware+'_infer')
    allure.dynamic.description('模型预测')
    r = re.search('/(.*)/', yml_name)
    category=r.group(1)
    print(category)
    model = TestOcrModelFunction(model=model_name, yml=yml_name, category=category)
    model.test_ocr_rec_infer(use_gpu)

@allure.story('export_model')
@pytest.mark.parametrize('yml_name', get_model_list())
@pytest.mark.parametrize("use_gpu", [True,False])
def test_rec_accuracy_export_model(yml_name, use_gpu):
    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    if use_gpu==True:
       hardware='_GPU'
    else:
       hardware='_CPU'
    allure.dynamic.title(model_name+hardware+'_export_model')
    allure.dynamic.description('模型动转静')
    r = re.search('/(.*)/', yml_name)
    category=r.group(1)
    print(category)
    model = TestOcrModelFunction(model=model_name, yml=yml_name, category=category)
    model.test_ocr_export_model(use_gpu)

@allure.story('predict')
@pytest.mark.parametrize('yml_name', get_model_list())
@pytest.mark.parametrize("enable_mkldnn", [True, False])
def test_rec_accuracy_predict_mkl(yml_name, enable_mkldnn):
    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    if enable_mkldnn==True:
       hardware='_mkldnn'
    else:
       hardware='_CPU'
    allure.dynamic.title(model_name+hardware+'_predict')
    allure.dynamic.description('预测库预测')
    r = re.search('/(.*)/', yml_name)
    category=r.group(1)
    print(category)
    model = TestOcrModelFunction(model=model_name, yml=yml_name, category=category)
    model.test_ocr_rec_predict(False, 0, enable_mkldnn)

@allure.story('predict')
@pytest.mark.parametrize('yml_name', get_model_list())
@pytest.mark.parametrize("use_tensorrt", [True,False])
def test_rec_accuracy_predict_trt(yml_name, use_tensorrt):
    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    if use_tensorrt==True:
       hardware='_tensorRT'
    else:
       hardware='_GPU'
    allure.dynamic.title(model_name+hardware+'_predict')
    allure.dynamic.description('预测库预测')
    r = re.search('/(.*)/', yml_name)
    category=r.group(1)
    print(category)
    model = TestOcrModelFunction(model=model_name, yml=yml_name, category=category)
    model.test_ocr_rec_predict(True, use_tensorrt, 0)

@allure.story('train')
@pytest.mark.parametrize('yml_name', get_model_list())
@pytest.mark.parametrize("use_gpu", [True])
def test_rec_funtion_train(yml_name, use_gpu):
    model_name=os.path.splitext(os.path.basename(yml_name))[0]
    if use_gpu==True:
       hardware='_GPU'
    else:
       hardware='_CPU'
    allure.dynamic.title(model_name+hardware+'_train')
    allure.dynamic.description('训练')
    r = re.search('/(.*)/', yml_name)
    category=r.group(1)
    print(category)
    model = TestOcrModelFunction(model=model_name, yml=yml_name, category=category)
    model.test_ocr_train(use_gpu)

