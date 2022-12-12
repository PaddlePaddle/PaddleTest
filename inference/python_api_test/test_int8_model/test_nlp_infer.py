"""
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
"""

import argparse
import os
import time
import sys
from functools import partial
import distutils.util

import numpy as np

import paddle
from paddle import inference
from paddle.metric import Metric, Accuracy, Precision, Recall
from paddlenlp.transformers import AutoModelForTokenClassification, AutoTokenizer
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.metrics import Mcc, PearsonAndSpearman
from backend import PaddleInferenceEngine, TensorRTEngine, ONNXRuntimeEngine

METRIC_CLASSES = {
    "cola": Mcc,
    "sst-2": Accuracy,
    "sts-b": PearsonAndSpearman,
    "mnli": Accuracy,
    "qnli": Accuracy,
    "rte": Accuracy,
    "afqmc": Accuracy,
    "tnews": Accuracy,
    "iflytek": Accuracy,
    "ocnli": Accuracy,
    "cmnli": Accuracy,
    "cluewsc2020": Accuracy,
    "csl": Accuracy,
}


def argsparser():
    """
    argsparser func
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="./afqmc",
        type=str,
        required=True,
        help="The path prefix of inference model to be used.",
    )
    parser.add_argument("--model_filename", type=str, default="inference.pdmodel", help="model file name")
    parser.add_argument("--params_filename", type=str, default="inference.pdiparams", help="params file name")
    parser.add_argument(
        "--task_name",
        default="afqmc",
        type=str,
        help="The name of the task to perform predict, selected in the list: " + ", ".join(METRIC_CLASSES.keys()),
    )
    parser.add_argument(
        "--dataset",
        default="clue",
        type=str,
        help="The dataset of model.",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size for predict.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--perf_warmup_steps",
        default=20,
        type=int,
        help="Warmup steps for performance test.",
    )
    parser.add_argument(
        "--use_trt",
        action="store_true",
        help="Whether to use inference engin TensorRT.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "int8"],
        help="The precision of inference. It can be 'fp32', 'fp16' or 'int8'. Default is 'fp16'.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="GPU",
        help="Choose the device you want to run, it can be: CPU/GPU/XPU, default is GPU",
    )
    parser.add_argument("--use_dynamic_shape", type=bool, default=True, help="Whether use dynamic shape or not.")
    parser.add_argument("--use_mkldnn", type=bool, default=False, help="Whether use mkldnn or not.")
    parser.add_argument("--cpu_threads", type=int, default=1, help="Num of cpu threads.")
    parser.add_argument(
        "--deploy_backend",
        type=str,
        default="paddle_inference",
        help="deploy backend, it can be: `paddle_inference`, `tensorrt`, `onnxruntime`",
    )
    parser.add_argument("--calibration_file", type=str, default=None, help="quant onnx model calibration cache file.")
    parser.add_argument("--model_name", type=str, default="", help="model_name for benchmark")
    return parser


def _convert_example(example, dataset, tokenizer, label_list, max_seq_length=512):
    assert dataset in ["glue", "clue"], "This demo only supports for dataset glue or clue"
    """Convert a glue example into necessary features."""
    if dataset == "glue":
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # Get the label
        label = example["labels"]
        label = np.array([label], dtype=label_dtype)
        # Convert raw text to feature
        example = tokenizer(example["sentence"], max_seq_len=max_seq_length)

        return example["input_ids"], example["token_type_ids"], label

    else:  # if dataset == 'clue':
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # Get the label
        example["label"] = np.array(example["label"], dtype="int64").reshape((-1, 1))
        label = example["label"]
        # Convert raw text to feature
        if "keyword" in example:  # CSL
            sentence1 = " ".join(example["keyword"])
            example = {"sentence1": sentence1, "sentence2": example["abst"], "label": example["label"]}
        elif "target" in example:  # wsc
            text, query, pronoun, query_idx, pronoun_idx = (
                example["text"],
                example["target"]["span1_text"],
                example["target"]["span2_text"],
                example["target"]["span1_index"],
                example["target"]["span2_index"],
            )
            text_list = list(text)
            assert text[pronoun_idx : (pronoun_idx + len(pronoun))] == pronoun, "pronoun: {}".format(pronoun)
            assert text[query_idx : (query_idx + len(query))] == query, "query: {}".format(query)
            if pronoun_idx > query_idx:
                text_list.insert(query_idx, "_")
                text_list.insert(query_idx + len(query) + 1, "_")
                text_list.insert(pronoun_idx + 2, "[")
                text_list.insert(pronoun_idx + len(pronoun) + 2 + 1, "]")
            else:
                text_list.insert(pronoun_idx, "[")
                text_list.insert(pronoun_idx + len(pronoun) + 1, "]")
                text_list.insert(query_idx + 2, "_")
                text_list.insert(query_idx + len(query) + 2 + 1, "_")
            text = "".join(text_list)
            example["sentence"] = text
        if tokenizer is None:
            return example
        if "sentence" in example:
            example = tokenizer(example["sentence"], max_seq_len=max_seq_length)
        elif "sentence1" in example:
            example = tokenizer(example["sentence1"], text_pair=example["sentence2"], max_seq_len=max_seq_length)
        return example["input_ids"], example["token_type_ids"], label


class WrapperPredictor(object):
    """
    Inference Predictor class
    """

    def __init__(self, predictor):
        self.predictor = predictor

    def predict_batch(self, data):
        """
        predict from batch func
        """
        self.predictor.prepare_data(data)
        output = self.predictor.run()
        return output

    def _convert_predict_batch(self, FLAGS, data, tokenizer, batchify_fn, label_list):
        examples = []
        for example in data:
            example = _convert_example(
                example, FLAGS.dataset, tokenizer, label_list, max_seq_length=FLAGS.max_seq_length
            )
            examples.append(example)

        return examples

    def eval(self, dataset, tokenizer, batchify_fn, FLAGS):
        """
        predict func
        """
        batches = [dataset[idx : idx + FLAGS.batch_size] for idx in range(0, len(dataset), FLAGS.batch_size)]

        for i, batch in enumerate(batches):
            examples = self._convert_predict_batch(FLAGS, batch, tokenizer, batchify_fn, dataset.label_list)
            input_ids, segment_ids, label = batchify_fn(examples)
            output = self.predict_batch([input_ids, segment_ids])
            if i > FLAGS.perf_warmup_steps:
                break

        metric = METRIC_CLASSES[FLAGS.task_name]()
        metric.reset()
        predict_time = 0.0
        for i, batch in enumerate(batches):
            examples = self._convert_predict_batch(FLAGS, batch, tokenizer, batchify_fn, dataset.label_list)
            input_ids, segment_ids, label = batchify_fn(examples)
            start_time = time.time()
            output = self.predict_batch([input_ids, segment_ids])
            end_time = time.time()
            predict_time += end_time - start_time
            correct = metric.compute(paddle.to_tensor(output), paddle.to_tensor(np.array(label).flatten()))
            metric.update(correct)

        sequences_num = i * FLAGS.batch_size
        print(
            "[benchmark]task name: {}, batch size: {} Inference time per batch: {}ms, qps: {}.".format(
                FLAGS.task_name,
                FLAGS.batch_size,
                round(predict_time * 1000 / i, 2),
                round(sequences_num / predict_time, 2),
            )
        )
        res = metric.accumulate()
        print("[benchmark]task name: %s, acc: %s. \n" % (FLAGS.task_name, res), end="")
        final_res = {
            "model_name": FLAGS.model_name,
            "jingdu": {
                "value": res,
                "unit": "acc",
            },
            "xingneng": {
                "value": round(predict_time * 1000 / i, 2),
                "unit": "ms",
                "batch_size": FLAGS.batch_size,
            },
        }
        print("[Benchmark][final result]{}".format(final_res))
        sys.stdout.flush()


def main(FLAGS):
    """
    main func
    """
    paddle.seed(42)
    task_name = FLAGS.task_name.lower()
    if FLAGS.use_mkldnn:
        paddle.set_device("cpu")
    token_dir = FLAGS.model_path
    predictor = None
    if FLAGS.deploy_backend == "paddle_inference":
        predictor = PaddleInferenceEngine(
            model_dir=FLAGS.model_path,
            model_filename=FLAGS.model_filename,
            params_filename=FLAGS.params_filename,
            precision=FLAGS.precision,
            use_trt=FLAGS.use_trt,
            use_mkldnn=FLAGS.use_mkldnn,
            batch_size=FLAGS.batch_size,
            device=FLAGS.device,
            min_subgraph_size=3,
            use_dynamic_shape=FLAGS.use_dynamic_shape,
            cpu_threads=FLAGS.cpu_threads,
        )
    elif FLAGS.deploy_backend == "tensorrt":
        model_name = os.path.split(FLAGS.model_path)[-1].rstrip(".onnx")
        token_dir = os.path.dirname(FLAGS.model_path)
        engine_file = "{}_{}_model.trt".format(model_name, FLAGS.precision)
        print(engine_file)
        predictor = TensorRTEngine(
            onnx_model_file=FLAGS.model_path,
            shape_info={
                "input_ids": [[28, 37], [32, 51], [32, 128]],
                "token_type_ids": [[28, 37], [32, 51], [32, 128]],
            },
            max_batch_size=FLAGS.batch_size,
            precision=FLAGS.precision,
            engine_file_path=engine_file,
            calibration_cache_file=FLAGS.calibration_file,
            verbose=False,
        )
    elif FLAGS.deploy_backend == "onnxruntime":
        model_name = os.path.split(FLAGS.model_path)[-1].rstrip(".onnx")
        token_dir = os.path.dirname(FLAGS.model_path)
        engine_file = "{}_{}_model.trt".format(model_name, FLAGS.precision)
        predictor = ONNXRuntimeEngine(
            onnx_model_file=FLAGS.model_path,
            precision=FLAGS.precision,
            use_trt=FLAGS.use_trt,
            use_mkldnn=FLAGS.use_mkldnn,
            device=FLAGS.device,
        )
    dev_ds = load_dataset("clue", task_name, splits="dev")
    tokenizer = AutoTokenizer.from_pretrained(token_dir)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # segment
        Stack(dtype="int64" if dev_ds.label_list else "float32"),  # label
    ): fn(samples)

    WrapperPredictor(predictor).eval(dev_ds, tokenizer, batchify_fn, FLAGS)
    rerun_flag = True if hasattr(predictor, "rerun_flag") and predictor.rerun_flag else False
    if rerun_flag:
        print("***** Collect dynamic shape done, Please rerun the program to get correct results. *****")


if __name__ == "__main__":
    # If the device is not set to cpu, the nv-trt will report an error when executing
    paddle.set_device("cpu")
    parser = argsparser()
    FLAGS = parser.parse_args()
    main(FLAGS)
