#!/usr/bin/env bash

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
####################################
export python=$1
export paddle=$2
export nlp_dir=/workspace/PaddleNLP
mkdir /workspace/PaddleNLP/model_logs
export log_path=/workspace/PaddleNLP/model_logs
export case_list=()

####################################
# Insatll paddlepaddle-gpu
install_paddle(){
    echo -e "\033[35m ---- Install paddlepaddle-gpu  \033[0m"
    python -m pip install --user ${paddle} --forceoreinstall --no-dependencies;
    python -c "import paddle; print('paddle version:',paddle.__version__,'\npaddle commit:',paddle.version.commit)";
}
####################################
get_diff_TO_case(){
for file_name in `git diff --numstat upstream/${AGILE_COMPILE_BRANCH} |awk '{print $NF}'`;do
    arr_file_name=(${file_name//// })
    dir1=${arr_file_name[0]}
    dir2=${arr_file_name[1]}
    dir3=${arr_file_name[2]}
    echo "file_name:"${file_name}, 
    echo "dir1:"${dir1}, "dir2:"${dir2},"dir3:"${dir3},".xx:" ${file_name##*.}
    if [ ! -f ${file_name} ];then # 针对pr删掉文件
        continue
    elif [[ ${file_name##*.} == "md" ]] || [[ ${file_name##*.} == "rst" ]] || [[ ${dir1} == "docs" ]];then
        continue
    elif [[ ${dir0} =~ "python" ]] && [[ ${dir1} =~ "paddle" ]];then
        if [[ ${dir2} =~ "distributed" ]] || [[ ${dir2} =~ "fluid" ]];then
            # python/paddle/distributed  || python/paddle/fluid
            case_list[${#case_list[*]}]=auto_parallel
        else
            continue
        fi
    elif [[ ${dir0} =~ "python" ]] && [[ ${dir1} =~ "fluid" ]];then
        if [[ ${dir2} =~ "distributed" ]];then
            # paddle/fluid/distributed
            case_list[${#case_list[*]}]=auto_parallel
        elif [[ ${dir2} =~ "framework" ]] && [[ ${dir3} =~ "new_executor" ]];then
            # paddle/fluid/framework/new_executor
            case_list[${#case_list[*]}]=auto_parallel
        else
            continue
        fi
    else
        continue
    fi
done
}
####################################
print_info(){
if [ $1 -ne 0 ];then
    mv ${log_path}/$2 ${log_path}/$2_FAIL.log
    echo -e "\033[31m ${log_path}/$2_FAIL \033[0m"
    cat ${log_path}/$2_FAIL.log
else
    echo -e "\033[32m ${log_path}/$2_SUCCESS \033[0m"
fi
}
####################################
get_diff_TO_case # 获取待执行case列表
case_list=($(awk -v RS=' ' '!a[$1]++' <<< ${case_list[*]}))  # 去重并将结果存储回原列表
if [[ ${#case_list[*]} -ne 0 ]];then
    # Install paddle
    install_paddle
    echo -e "\033[35m =======CI Check case========= \033[0m"
    echo -e "\033[35m ---- case_list length: ${#case_list[*]}, cases: ${case_list[*]} \033[0m"
    set +e
    echo -e "\033[35m ---- start run case  \033[0m"
    case_num=1
    for case in ${case_list[*]};do
        echo -e "\033[35m ---- running case $case_num/${#case_list[*]}: ${case} \033[0m"
        bash ./ci_case.sh ${case} 
        print_info $? `ls -lt ${log_path} | grep gpt | head -n 1 | awk '{print $9}'`
        let case_num++
        fi
    done
    echo -e "\033[35m ---- end run case  \033[0m"
    cd ${nlp_dir}/model_logs
    FF=`ls *FAIL*|wc -l`
    EXCODE=0
    if [ "${FF}" -gt "0" ];then
        case_EXCODE=1
        EXCODE=2
    else
        case_EXCODE=0
    fi
    if [ $case_EXCODE -ne 0 ] ; then
        echo -e "\033[31m ---- case Failed number: ${FF} \033[0m"
        ls *_FAIL*
    else
        echo -e "\033[32m ---- case Success \033[0m"
    fi
else
    echo -e "\033[32m Changed Not CI case, Skips \033[0m"
    EXCODE=0
fi
exit $EXCODE

