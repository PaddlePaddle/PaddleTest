#!/bin/bash

config_file=$1
mode=$2

# skip other modes
if [ ${mode} != "lite_train_lite_infer" ]; then
    echo "upload model only in lite_train_lite_infer"
    exit 0
fi

# prepare SDK
push_dir=$PWD
push_file=${push_dir}/bce-python-sdk-0.8.27/BosClient.py
cp /workspace/bce_whl.tar.gz ./
if [ ! -f ${push_dir}/bce_whl.tar.gz ];then
    echo "BOS SDK pull failed"
    exit 1
fi
if [ ! -f ${push_file} ];then
    tar xf $PWD/bce_whl.tar.gz -C ${push_dir}
fi

# get model info
time_stamp=`date +%Y_%m_%d`
cd "/workspace/${REPO}"
repo_commit=`git rev-parse HEAD`
cd -
cd /workspace
paddle_commit=`git rev-parse HEAD`
cd -
model_name=`cat ${config_file} | grep model_name | awk -F ":" '{print $NF}' | head -n 1`
echo ${model_name}
output_dir="test_tipc/output/norm_train_gpus_0,1_autocast_null/${model_name}"
# special model
if [ ${model_name} == "GeneralRecognition_PPLCNet_x2_5" ]; then
    output_dir="test_tipc/output/norm_train_gpus_0,1_autocast_null/RecModel"
fi
echo ${output_dir}
if [ ! -d ${output_dir} ]; then
    echo "output_dir not found"
    exit 1
fi
# copy model files
upload_dir="${output_dir}_upload"
mkdir -p ${upload_dir}
\cp test_tipc/output/norm_train_gpus_0,1_autocast_null/inference.pdiparams ${upload_dir}
\cp test_tipc/output/norm_train_gpus_0,1_autocast_null/inference.pdiparams.info ${upload_dir}
\cp test_tipc/output/norm_train_gpus_0,1_autocast_null/inference.pdmodel ${upload_dir}
\cp ${config_file} ${upload_dir}

# upload model
model_tar_name="${time_stamp}^${REPO}^${model_name}^${paddle_commit}^${repo_commit}.tgz"
models_link_file="tipc_models_url.txt"
model_url="https://paddle-qa.bj.bcebos.com/fullchain_ce_test/${model_tar_name}"
path_suffix=${upload_dir##*/}
tar -zcvf ${model_tar_name} -C test_tipc/output/norm_train_gpus_0,1_autocast_null ${path_suffix}
python2 ${push_file} ${model_tar_name} paddle-qa/fullchain_ce_test
exit_code=$?
if [ ${exit_code} == 0 ]; then
    echo ${model_url} >> ${models_link_file}
    python2 ${push_file} ${models_link_file} paddle-qa/fullchain_ce_test/model_download_link
fi
rm -rf ${model_tar_name}
rm -rf ${upload_dir}
exit $exit_code
