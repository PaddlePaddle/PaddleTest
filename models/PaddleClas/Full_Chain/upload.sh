#!/bin/bash

config_file=$1

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
model_name=`cat ${config_file} | grep model_name | awk -F ":" '{print $NF}'`
echo ${model_name}
output_dir="test_tipc/output/norm_train_gpus_0,1_autocast_null/${model_name}"
echo ${output_dir}
if [ ! -d ${output_dir} ]; then
    echo "output_dir not found"
    exit 1
fi
# copy model files
\cp test_tipc/output/norm_train_gpus_0,1_autocast_null/inference.pdiparams ${output_dir}
\cp test_tipc/output/norm_train_gpus_0,1_autocast_null/inference.pdiparams.info ${output_dir}
\cp test_tipc/output/norm_train_gpus_0,1_autocast_null/inference.pdmodel ${output_dir}
\cp ${config_file} ${output_dir}

# upload model
model_tar_name="${time_stamp}^${REPO}^${model_name}^${paddle_commit}^${repo_commit}.tgz"
path_suffix=${output_dir##*/}
tar -zcvf ${model_tar_name} -C test_tipc/output/norm_train_gpus_0,1_autocast_null ${path_suffix}
python2 ${push_file} ${model_tar_name} paddle-qa/fullchain_ce_test
exit_code=$?
rm -rf ${model_tar_name}
exit $exit_code
