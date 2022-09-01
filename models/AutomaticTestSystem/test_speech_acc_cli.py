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
    with open(filename, encoding='utf-8') as f:
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

