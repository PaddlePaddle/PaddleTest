#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
  * @file
  * @author jiaxiao01
  * @date 2021/9/3 3:46 PM
  * @brief clas model inference test case
  *
  **************************************************************************/
"""

import subprocess
import re
import pytest
import numpy as np

from RocmTestFramework import TestNlpModel
from RocmTestFramework import RepoInitCustom
from RocmTestFramework import RepoRemove
from RocmTestFramework import RepoDataset
from RocmTestFramework import clean_process
from RocmTestFramework import dependency_install


def setup_module():
    """
    setup
    """
    RepoInitCustom(repo="PaddleNLP")
    RepoDataset(
        cmd="python -m pip install sentencepiece -i https://pypi.tuna.tsinghua.edu.cn/simple; \
             python -m pip install attrdict -i https://pypi.tuna.tsinghua.edu.cn/simple;"
    )
    dependency_install(package="paddlenlp")


def teardown_module():
    """
    teardown
    """
    RepoRemove(repo="PaddleNLP")


def setup_function():
    """
    clean_process
    """
    clean_process()


def test_bert_pretrain():
    """
    bert_pretrain
    """
    model = TestNlpModel(directory="examples/language_model/bert/")
    cmd = """cd PaddleNLP/examples/language_model/bert/; \
             python create_pretraining_data.py --input_file=data/sample_text.txt \
--output_file=data/training_data.hdf5 --bert_model=bert-base-uncased --max_seq_length=128 \
--max_predictions_per_seq=20 --masked_lm_prob=0.15 --random_seed=12345 --dupe_factor=5; \
              python -m paddle.distributed.launch --gpus "0,1,2,3" run_pretrain.py \
--model_type bert --model_name_or_path bert-base-uncased --max_predictions_per_seq 20 --batch_size 32 \
 --learning_rate 1e-4  --weight_decay 1e-2 --adam_epsilon 1e-6 --warmup_steps 10000 --num_train_epochs 1 \
--input_dir data/   --output_dir pretrained_models/ --logging_steps 1  --save_steps 20000 --max_steps 1000000 \
--device gpu --use_amp False"""
    model.test_nlp_train(cmd=cmd)


def test_bert_finetune():
    """
    bert_finetune
    """
    model = TestNlpModel(directory="examples/language_model/bert/")
    cmd = """cd PaddleNLP/examples/language_model/bert/; \
           export HIP_VISIBLE_DEVICES=0,1,2,3; \
           python -W ignore -m paddle.distributed.launch --gpus "0,1,2,3" --log_dir log_SST-2_gpu4 ./run_glue.py  \
--model_type bert --model_name_or_path bert-base-cased --task_name SST-2 --max_seq_length 128 --batch_size 32 \
--learning_rate 1e-4 --num_train_epochs 1 --logging_steps 10 --save_steps 500 \
--output_dir ./output_SST-2_gpu4/ --device gpu
        """
    model.test_nlp_train(cmd=cmd)


def test_xlnet():
    """
    xlnet
    """
    model = TestNlpModel(directory="examples/language_model/xlnet/")
    cmd = """cd PaddleNLP/examples/language_model/xlnet/;
           export HIP_VISIBLE_DEVICES=0,1,2,3
           python -W ignore -m paddle.distributed.launch --gpus "0,1,2,3" --log_dir log_SST-2_gpu4 ./run_glue.py  \
--model_name_or_path xlnet-base-cased --task_name SST-2 --max_seq_length 128 --batch_size 32 --learning_rate 2e-5 \
--num_train_epochs 1 --logging_steps 100 --save_steps 500 --output_dir ./output_SST-2_gpu4/ --device gpu
        """
    model.test_nlp_train(cmd=cmd)


def test_electra():
    """
    electra
    """
    model = TestNlpModel(directory="examples/language_model/electra/")
    cmd = """cd PaddleNLP/examples/language_model/electra/;
           export HIP_VISIBLE_DEVICES=0,1,2,3
           python -W ignore -m paddle.distributed.launch --gpus "0,1,2,3" --log_dir log_SST-2_gpu4 ./run_glue.py  \
--model_type electra --model_name_or_path electra-small --task_name SST-2 --max_seq_length 128 --batch_size 32 \
--learning_rate 1e-4 --num_train_epochs 1 --logging_steps 10 --save_steps 500 --output_dir ./output_SST-2_gpu4/ \
--device gpu
        """
    model.test_nlp_train(cmd=cmd)


def test_transformer():
    """
    transformer
    """
    model = TestNlpModel(directory="examples/machine_translation/transformer/")
    cmd = """cd PaddleNLP/examples/machine_translation/transformer/; \
             python -m paddle.distributed.launch --gpus "0,1,2,3" train.py \
--config ./configs/transformer.base.yaml --max_iter 10"""
    model.test_nlp_train(cmd=cmd)


@pytest.mark.skip(reason="seq2seq train/eval too long, maybe hang")
def test_seq2seq():
    """
    seq2seq
    """
    model = TestNlpModel(directory="examples/machine_translation/seq2seq/")
    cmd = """cd PaddleNLP/examples/machine_translation/seq2seq/; \
           export HIP_VISIBLE_DEVICES=0,1,2,3; \
           python -m paddle.distributed.launch --gpus="0,1,2,3"  train.py --num_layers 2 \
--hidden_size 512 --batch_size 128 --dropout 0.2 --init_scale  0.1 --max_grad_norm 5.0  \
--device gpu --model_path ./attention_models --max_epoch 1"""
    model.test_nlp_train(cmd=cmd)


def test_textcnn():
    """
    textcnn
    """
    model = TestNlpModel(directory="examples/text_classification/rnn")
    cmd = """cd PaddleNLP/examples/text_classification/rnn; \
           wget https://paddlenlp.bj.bcebos.com/data/senta_word_dict.txt; \
           export HIP_VISIBLE_DEVICES=0,1,2,3; \
           python -m paddle.distributed.launch --gpus="0,1,2,3" train.py \
--vocab_path=./senta_word_dict.txt --device=gpu  --network=cnn --lr=1e-4 --batch_size=64 \
--epochs=1 --save_dir=./checkpoints"""
    model.test_nlp_train(cmd=cmd)


def test_rnn():
    """
    rnn
    """
    model = TestNlpModel(directory="examples/text_classification/rnn")
    cmd = """cd PaddleNLP/examples/text_classification/rnn;
           wget https://paddlenlp.bj.bcebos.com/data/senta_word_dict.txt
           export HIP_VISIBLE_DEVICES=0,1,2,3
           python -m paddle.distributed.launch --gpus="0,1,2,3" train.py --vocab_path=./senta_word_dict.txt \
--device=gpu  --network=bilstm --lr=5e-4 --batch_size=64 --epochs=1 --save_dir=./checkpoints"""
    model.test_nlp_train(cmd=cmd)
