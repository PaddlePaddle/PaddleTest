# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import io
import os
import sys
import subprocess
import json


def exit_check(
    exit_code,
    file_name,
):
    """
    check exit_code
    """
    assert exit_code == 0, "%s Failed!" % (file_name)


def save_log(exit_code, output, file_name):
    """ """
    if not os.path.exists("test_log"):
        os.mkdir("test_log")
    if exit_code == 0:
        log_file = os.getcwd() + "/test_log/" + os.path.join(file_name + "_success.log")
        with open(log_file, "a") as flog:
            flog.write("%s" % (output))
    else:
        log_file = os.getcwd() + "/test_log/" + os.path.join(file_name + "_err.log")
        with open(log_file, "a") as flog:
            flog.write("%s" % (output))


def test_requirements():
    """"""
    file_name = "requirements"
    if os.path.exists("requirements.txt"):
        cmd = f"pip install -r requirements.txt"
        output = subprocess.getstatusoutput(cmd)
        save_log(output[0], output[1], file_name)
        exit_check(output[0], file_name)


def test_prepare_data_files():
    """"""
    file_name = "prepare_data_files"
    if not os.path.exists("pre_data"):
        os.mkdir("pre_data")
    cmd = """cd ./pre_data &&
    wget -q https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_ids.npy &&
    wget -q https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_idx.npz && cd ../"""
    output = subprocess.getstatusoutput(cmd)
    save_log(output[0], output[1], file_name)
    exit_check(output[0], file_name)


def test_run_pretrain():
    """"""
    file_name = "run_pretrain"
    cmd = """python -m paddle.distributed.launch run_pretrain.py \
    --model_type gpt \
    --model_name_or_path gpt2-en \
    --input_dir "./pre_data" \
    --output_dir "output" \
    --weight_decay 0.01 \
    --grad_clip 1.0 \
    --max_steps 2 \
    --save_steps 2 \
    --decay_steps 320000 \
    --warmup_rate 0.01 \
    --micro_batch_size 2 \
    --device gpu"""
    output = subprocess.getstatusoutput(cmd)
    save_log(output[0], output[1], file_name)
    exit_check(output[0], file_name)


def test_run_eval():
    """"""
    file_name = "run_eval"
    get_data_cmd = (
        "wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip && unzip wikitext-103-v1.zip"
    )
    os.system(get_data_cmd)
    cmd = """python run_eval.py --model_name gpt2-en \
    --eval_path ./wikitext-103/wiki.valid.tokens \
    --overlapping_eval 32 \
    --init_checkpoint_path ./output/model_2/model_state.pdparams \
    --batch_size 8 \
    --device gpu"""

    output = subprocess.getstatusoutput(cmd)
    save_log(output[0], output[1], file_name)
    exit_check(output[0], file_name)


def test_export_model():
    """"""
    file_name = "export_model"
    cmd = """python export_model.py --model_type=gpt-cn \
    --model_path=gpt-cpm-large-cn \
    --output_path=./infer_model/model"""

    output = subprocess.getstatusoutput(cmd)
    save_log(output[0], output[1], file_name)
    exit_check(output[0], file_name)


def test_inference():
    """"""
    file_name = "inference"
    cmd = """python deploy/python/inference.py --model_type gpt-cn \
    --model_path ./infer_model/model"""

    output = subprocess.getstatusoutput(cmd)
    save_log(output[0], output[1], file_name)
    exit_check(output[0], file_name)
