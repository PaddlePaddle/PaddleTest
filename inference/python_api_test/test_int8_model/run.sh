#! /bin/bash

bash prepare.sh

mv models models.bak
cp -r models.bak models

bash run_trt_int8.sh > eval_trt_int8_acc.log.tmp 2>&1
bash run_trt_int8.sh > eval_trt_int8_acc.log 2>&1

rm -rf models
cp -r models.bak models

bash run_trt_fp16.sh > eval_trt_fp16_acc.log.tmp 2>&1
bash run_trt_fp16.sh > eval_trt_fp16_acc.log 2>&1

rm -rf models
cp -r models.bak models

bash run_mkldnn_int8.sh > eval_mkldnn_int8_acc.log 2>&1

bash run_mkldnn_fp32.sh > eval_mkldnn_fp32_acc.log 2>&1

python get_benchmark_info.py
