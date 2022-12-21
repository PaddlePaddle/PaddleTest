"""
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
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.metrics import AccuracyAndF1, Mcc, PearsonAndSpearman
from paddlenlp.transformers import BertForSequenceClassification, BertTokenizer
from backend import PaddleInferenceEngine, TensorRTEngine, ONNXRuntimeEngine, Monitor

METRIC_CLASSES = {
    "cola": Mcc,
    "sst-2": Accuracy,
    "mrpc": AccuracyAndF1,
    "sts-b": PearsonAndSpearman,
    "qqp": AccuracyAndF1,
    "mnli": Accuracy,
    "qnli": Accuracy,
    "rte": Accuracy,
}

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("sentence1", "sentence2"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("sentence1", "sentence2"),
    "qqp": ("sentence1", "sentence2"),
    "rte": ("sentence1", "sentence2"),
    "sst-2": ("sentence", None),
    "sts-b": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def argsparser():
    """
    parse_args func
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="./x2paddle_cola",
        type=str,
        required=True,
        help="The path prefix of inference model to be used.",
    )
    parser.add_argument("--model_filename", type=str, default="model.pdmodel", help="model file name")
    parser.add_argument("--params_filename", type=str, default="model.pdiparams", help="params file name")
    parser.add_argument(
        "--task_name",
        default="cola",
        type=str,
        help="The name of the task to perform predict, selected in the list: " + ", ".join(METRIC_CLASSES.keys()),
    )
    parser.add_argument("--model_type", default="bert-base-cased", type=str, help="Model type selected in bert.")
    parser.add_argument(
        "--model_name_or_path",
        default="bert-base-cased",
        type=str,
        help="The directory or name of model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="GPU",
        help="Choose the device you want to run, it can be: CPU/GPU/XPU, default is GPU",
    )
    parser.add_argument("--use_dynamic_shape", type=bool, default=True, help="Whether use dynamic shape or not.")
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


def _convert_example(
    example,
    tokenizer,
    label_list,
    max_seq_length=512,
    task_name=None,
    is_test=False,
    padding="max_length",
    return_attention_mask=True,
):
    if not is_test:
        # `label_list == None` is for regression task
        label_dtype = "int64" if label_list else "float32"
        # Get the label
        label = example["labels"]
        label = np.array([label], dtype=label_dtype)
    # Convert raw text to feature
    sentence1_key, sentence2_key = task_to_keys[task_name]
    texts = (example[sentence1_key],) if sentence2_key is None else (example[sentence1_key], example[sentence2_key])
    example = tokenizer(
        *texts, max_seq_len=max_seq_length, padding=padding, return_attention_mask=return_attention_mask
    )
    if not is_test:
        if return_attention_mask:
            return example["input_ids"], example["attention_mask"], example["token_type_ids"], label
        else:
            return example["input_ids"], example["token_type_ids"], label
    else:
        if return_attention_mask:
            return example["input_ids"], example["attention_mask"], example["token_type_ids"]
        else:
            return example["input_ids"], example["token_type_ids"]


class WrapperPredictor(object):
    """
    Inference Predictor class
    """

    def __init__(self, predictor):
        self.predictor = predictor

    def eval(self, dataset, collate_fn, FLAGS):
        """
        predict func
        """
        batch_sampler = paddle.io.BatchSampler(dataset, batch_size=FLAGS.batch_size, shuffle=False)
        data_loader = paddle.io.DataLoader(
            dataset=dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, num_workers=0, return_list=True
        )

        for i, data in enumerate(data_loader):
            data = [ele.numpy() if isinstance(ele, paddle.Tensor) else ele for ele in data]
            real_data = data[0:3]
            self.predictor.prepare_data(real_data)
            output = self.predictor.run()
            if i > FLAGS.perf_warmup_steps:
                break

        metric = METRIC_CLASSES[FLAGS.task_name]()
        metric.reset()
        predict_time = 0.0

        monitor = Monitor(0)
        monitor.start()
        for i, data in enumerate(data_loader):
            data = [ele.numpy() if isinstance(ele, paddle.Tensor) else ele for ele in data]
            real_data = data[0:3]
            self.predictor.prepare_data(real_data)
            start_time = time.time()
            output = self.predictor.run()
            end_time = time.time()
            predict_time += end_time - start_time
            label = data[-1]
            correct = metric.compute(paddle.to_tensor(output[0]), paddle.to_tensor(np.array(label).flatten()))
            metric.update(correct)

        monitor.stop()
        monitor_result = monitor.output()

        cpu_mem = (
            monitor_result["result"]["cpu_memory.used"]
            if ("result" in monitor_result and "cpu_memory.used" in monitor_result["result"])
            else 0
        )
        gpu_mem = (
            monitor_result["result"]["gpu_memory.used"]
            if ("result" in monitor_result and "gpu_memory.used" in monitor_result["result"])
            else 0
        )

        print("[Benchmark] cpu_mem:{} MB, gpu_mem: {} MB".format(cpu_mem, gpu_mem))

        sequences_num = i * FLAGS.batch_size
        print(
            "[Benchmark]task name: {}, batch size: {} Inference time per batch: {}ms, qps: {}.".format(
                FLAGS.task_name,
                FLAGS.batch_size,
                round(predict_time * 1000 / i, 2),
                round(sequences_num / predict_time, 2),
            )
        )
        res = metric.accumulate()
        print("[Benchmark]task name: %s, acc: %s. \n" % (FLAGS.task_name, res), end="")
        final_res = {
            "model_name": FLAGS.model_name,
            "jingdu": {
                "value": res[0],
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
    predictor = None
    token_dir = FLAGS.model_path
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
            min_subgraph_size=5,
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
                "x0": [[1, 128], [1, 128], [1, 128]],
                "x1": [[1, 128], [1, 128], [1, 128]],
                "x2": [[1, 128], [1, 128], [1, 128]],
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
    FLAGS.task_name = FLAGS.task_name.lower()
    FLAGS.model_type = FLAGS.model_type.lower()

    dev_ds = load_dataset("glue", FLAGS.task_name, splits="dev")
    print(FLAGS.model_name_or_path)
    tokenizer = BertTokenizer.from_pretrained(token_dir)

    trans_func = partial(
        _convert_example,
        tokenizer=tokenizer,
        label_list=dev_ds.label_list,
        max_seq_length=FLAGS.max_seq_length,
        task_name=FLAGS.task_name,
        return_attention_mask=True,
    )

    dev_ds = dev_ds.map(trans_func)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=0),
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # segment
        Stack(dtype="int64" if dev_ds.label_list else "float32"),  # label
    ): fn(samples)
    WrapperPredictor(predictor).eval(dev_ds, batchify_fn, FLAGS)


if __name__ == "__main__":
    # If the device is not set to cpu, the nv-trt will report an error when executing
    paddle.set_device("cpu")
    parser = argsparser()
    FLAGS = parser.parse_args()
    main(FLAGS)
