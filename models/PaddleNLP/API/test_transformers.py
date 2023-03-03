"""Test transformer API"""
import io
import re
import os
import warnings
import paddle
from paddlenlp.transformers.auto.modeling import *
from paddlenlp.transformers import *


def getModelList():
    """
    get model links from
    <https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/model_zoo/transformers/xx/contents.rst>
    """
    path = "./docs/model_zoo/transformers/"
    file_list = os.listdir(path)
    model_list = []
    keyword = ["|``", "| ``"]
    print(len(file_list), sorted(file_list))
    for file_name in file_list:
        with open(path + file_name + "/contents.rst") as f:
            for line in f:
                if any(word in line for word in keyword):
                    result = re.split(r"``", line)
                    model_list.append(result[1])
    return model_list


def test_transformers():
    """
    download PretrainedModels & PretrainedTokenizer
    """
    model_list = getModelList()
    count = 0
    save_dir = os.getenv("PPNLM_HOME") + "/test/icoding/autosave/"
    for model_name in model_list:
        count = count + 1
        print("processing:...{}/{}...".format(count, len(model_list)))
        try:
            num_classes = 1
            model_pretrained = AutoModel.from_pretrained(model_name, num_labels=num_classes)

            model_pretrained.save_pretrained(save_dir + model_name)
            model_local = AutoModel.from_pretrained(save_dir + model_name)
            model_state = paddle.load(save_dir + model_name + "/model_state.pdparams")
            model_pretrained.set_state_dict(model_state)

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # check special token
            tokenizer.pad_token
            tokenizer.unk_token

            # tokenizer.sep_token
            # tokenizer.cls_token
            # tokenizer.mask_token

            # tokenizer.bos_token
            # tokenizer.eos_token

            tokenizer.vocab_size

            # check tokenizer basic func
            token = tokenizer("china")
            token["input_ids"]
            token["token_type_ids"]

            tokenizer.save_pretrained(save_dir + model_name)
            tokenizer = AutoTokenizer.from_pretrained(save_dir + model_name)
        except Exception as e:
            with io.open("./transformer_error.log", "a+", encoding="utf-8") as flog:
                flog.write("{}: {}".format(model_name, e) + "\r\n")


if __name__ == "__main__":
    test_transformers()
