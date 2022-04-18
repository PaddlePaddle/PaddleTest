"""Test_transformer is used to check Auto API test case."""
import io
import re
import warnings
import paddle
from paddlenlp.transformers.auto.modeling import *
from paddlenlp.transformers import *


def getModelList():
    """
    get model links from
    <https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/model_zoo/transformers.rst>
    """
    model_list = []
    keyword = ["|``"]
    with open("../../../docs/model_zoo/transformers.rst") as f:
        for line in f:
            if any(word in line for word in keyword):
                reg = re.compile(r"\``+.*?\``")
                result = re.sub(r"``", "", reg.findall(line)[0])
                model_list.append(result)
        return model_list


def test_transformers():
    """
    download PretrainedModels & PretrainedTokenizer
    """
    model_list = getModelList()
    for model_temp in model_list:
        try:
            model = AutoModel.from_pretrained(model_temp)
            tokenizer = AutoTokenizer.from_pretrained(model_temp)
            with io.open("./transformer_success.log", "a+", encoding="utf-8") as flog:
                flog.write("Pretrained Weight:{}".format(model_temp) + "\r\n")
                flog.write("Model:{},tokenizer:{}".format(type(model), tokenizer) + "\r\n")
        except Exception as e:
            with io.open("./transformer_error.log", "a+", encoding="utf-8") as flog:
                flog.write("{}: {}".format(model_temp, e) + "\r\n")


if __name__ == "__main__":
    test_transformers()
