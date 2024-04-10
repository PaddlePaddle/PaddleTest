#!/bin/bash

export FLAGS_cudnn_exhaustive_search=1
export FLAGS_allocator_strategy=auto_growth
export CUDA_MODULE_LOADING=LAZY
export FLAGS_conv_workspace_size_limit=32
export FLAGS_initial_cpu_memory_in_mb=0
#export NVIDIA_TF32_OVERRIDE=0

model_dir=$PWD/Models/$1
rm -rf ${model_dir}/_opt*
rm -rf ${model_dir}/shape*
config_file=config.yaml

gpu_id=12
enable_gpu=false
enable_pir=false
enable_trt=false

if [ $1 == "mask_rcnn_r50_fpn_1x_coco" ]; then
  subgraph_size_var=8
else
  subgraph_size_var=3
fi
backend_type=$2
#if [ $2 == "1" ];then
#    backend_type=paddle
#else
#    backend_type=onnxruntime
#fi
if [ $3 == "fp32" ];then
    precision=fp32
else
    precision=fp16
fi
batch_size=$4

model_file=""
params_file=""
for file in $(ls $model_dir)
  do
    if [ "${file##*.}"x = "pdmodel"x ];then
      model_file=$file
      echo "find model file: $model_file"
    fi

    if [ "${file##*.}"x = "pdiparams"x ];then
      params_file=$file
      echo "find param file: $params_file"
    fi
done

#if [ $5 == "1" ];then
#    # auto tune
#    python benchmark.py --model_dir=${model_dir} --config_file ${config_file} --precision ${precision} --enable_gpu=${enable_gpu} --enable_pir=${enable_pir} --gpu_id=${gpu_id} --enable_trt=${enable_trt} --backend_type=${backend_type} --batch_size=${batch_size} --paddle_model_file "$model_file" --paddle_params_file "$params_file" --enable_tune=true --return_result=true
#fi
# infer
python benchmark.py --model_dir=${model_dir} --config_file ${config_file} --precision ${precision} --enable_gpu=${enable_gpu} --enable_pir=${enable_pir} --gpu_id=${gpu_id} --enable_trt=${enable_trt} --backend_type=${backend_type} --batch_size=${batch_size} --subgraph_size=${subgraph_size_var} --paddle_model_file "$model_file" --paddle_params_file "$params_file" --return_result=true
