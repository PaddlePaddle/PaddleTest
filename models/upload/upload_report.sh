#! /bin/bash
# $1:paddle_compile_path $2:repo_name $3:repo_branch $4:build_url $5:exit_code $6:status
echo ---export env variable for upload report---

complie_path=$1
path_temp=`echo ${complie_path} | awk -F 'paddlepaddle' '{print $1}'`
echo ${path_temp}
wget ${path_temp}description.txt
export build_commit_time=`grep commit_time description.txt | awk -F ':' '{print $2}'`

export build_type_id=${AGILE_PIPELINE_CONF_ID}
export build_id=${AGILE_PIPELINE_BUILD_ID}
export build_job_id=${AGILE_JOB_BUILD_ID}
export build_repo_name=$2
export build_repo_branch=$3
export build_commit_id=`python -c 'import paddle; print(paddle.version.commit)'`
export build_url=$4
export build_exit_code=$5
export build_status=$6

echo --print env variable---
export | grep build_

unset http_proxy && unset https_proxy
python upload_report.py
