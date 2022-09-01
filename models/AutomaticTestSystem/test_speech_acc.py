import pytest
import numpy as np
import subprocess
import re
import sys
import yaml
import platform
import os.path
import allure
import paddle
import random
import os

from ModelsTestFramework import RepoInitSpeech
from ModelsTestFramework import RepoDatasetSpeech
from ModelsTestFramework import TestSpeechModelFunction


def get_model_list(filename='models_list_speech.yaml'):
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
    RepoInitSpeech(repo='PaddleSpeech')
    RepoDatasetSpeech()

@allure.story('paddle_speech_cli')
@pytest.mark.parametrize('cmd', get_model_list('speech_cli_list.txt'))
def test_Speech_accuracy_cli(cmd):
    allure.dynamic.title('paddle_speech_cli')
    allure.dynamic.description('paddle_speech_cli')

    model = TestSpeechModelFunction()
    print(cmd)
    model.test_speech_cli(cmd)


@allure.story('get_pretrained_model')
@pytest.mark.parametrize('model_name', get_model_list())
def test_Speech_accuracy_get_pretrained_model(model_name):
    allure.dynamic.title(model_name+'_get_pretrained_model')
    allure.dynamic.description('获取预训练模型')

    model = TestSpeechModelFunction(model=model_name)
    model.test_speech_get_pretrained_model()

@allure.story('synthesize_e2e')
@pytest.mark.parametrize('model_name', get_model_list())
def test_speech_accuracy_synthesize_e2e(model_name):
    allure.dynamic.title(model_name+'_synthesize_e2e')
    allure.dynamic.description('模型评估')
    
    model = TestSpeechModelFunction(model=model_name)
    model.test_speech_synthesize_e2e()


@allure.story('train')
@pytest.mark.parametrize('model_name', get_model_list())
def test_speech_funtion_train(model_name):
    allure.dynamic.title(model_name+'_train')
    allure.dynamic.description('训练')

    model = TestSpeechModelFunction(model=model_name)
    model.test_speech_train()
