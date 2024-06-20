#!/bin/bash

function func_parser_value(){
    strs=$1
    IFS=":"
    array=(${strs})
    tmp=${array[1]}
    echo ${tmp}
}

function func_parser_dataset_url(){
    strs=$1
    IFS=":"
    array=(${strs})
    tmp=${array[2]}
    echo ${tmp}
}

function run_command(){
    command=$1
    module_name=$2
    time_stamp=$(date +"%Y-%m-%d %H:%M:%S")
    printf "\e[32m|%-20s| %-20s | %-50s | %-20s\n\e[0m" "[${time_stamp}]" ${module_name} "Run ${command}"
    eval $command
}

set -e

# 获取python的绝对路径
PYTHONPATH="python"
# 获取当前脚本的绝对路径，获得基准目录
BASE_PATH=$(cd "$(dirname $0)"; pwd)
MODULE_OUTPUT_PATH=${BASE_PATH}/outputs

install_deps_cmd="${PYTHONPATH} install_pdx.py -y"
eval ${install_deps_cmd}

IFS='*'
module_info_file=${BASE_PATH}/run_suite.txt
modules_info_list=($(cat ${module_info_file}))

for modules_info in ${modules_info_list[@]}; do
    IFS='='
    info_list=($modules_info)
    for module_info in ${info_list[@]}; do
        IFS=$'\n'
        if [[ $module_info == *check_dataset_yaml* ]]; then
            # 数据准备，获取模型信息和运行模式
            lines=(${module_info})
            module_name=$(func_parser_value "${lines[0]}")
            check_dataset_yaml=$(func_parser_value "${lines[1]}")
            dataset_url=https:$(func_parser_dataset_url "${lines[2]}")
            run_model=$(func_parser_value "${lines[3]}")
            check_options=$(func_parser_value "${lines[4]}")
            check_weights_items=$(func_parser_value "${lines[5]}")
            best_weight_path=$(func_parser_value "${lines[6]}")
            epochs_iters=$(func_parser_value "${lines[7]}")
            download_dataset_cmd="${PYTHONPATH} ${BASE_PATH}/checker.py --download_dataset --config_path ${check_dataset_yaml} --dataset_url ${dataset_url}"
            run_command ${download_dataset_cmd} ${module_name}
            model_output_path=${MODULE_OUTPUT_PATH}/${module_name}_dataset_check
            check_dataset_cmd="${PYTHONPATH} main.py -c ${check_dataset_yaml} -o Global.mode=check_dataset -o Global.output=${model_output_path}"
            run_command ${check_dataset_cmd} ${module_name}
            checker_cmd="${PYTHONPATH} ${BASE_PATH}/checker.py --check --check_dataset_result --output ${model_output_path} --module_name ${module_name}"
            run_command ${checker_cmd} ${module_name}

        elif [[ ! -z $module_info ]]; then
            for config_path in $module_info;do
                config_path=$(func_parser_value "${config_path}")
                model_output_path=${MODULE_OUTPUT_PATH}/${module_name}_output
                evaluate_weight_path=${model_output_path}/${best_weight_path}
                rm -rf ${model_output_path}
                IFS=$'|'
                run_model_list=(${run_model})
                for mode in ${run_model_list[@]};do
                    # 根据config运行各模型的train和evaluate
                    run_mode_cmd="${PYTHONPATH} main.py -c ${config_path} -o Global.mode=${mode} -o Global.output=${model_output_path} -o Train.epochs_iters=${epochs_iters} -o Evaluate.weight_path=${evaluate_weight_path}"
                    run_command ${run_mode_cmd} ${module_name}
                done
                check_options_list=(${check_options})
                for check_option in ${check_options_list[@]};do
                    # 运行产出检查脚本
                    checker_cmd="${PYTHONPATH} ${BASE_PATH}/checker.py --check --$check_option --output ${model_output_path} --check_weights_items ${check_weights_items} --module_name ${module_name}"
                    run_command ${checker_cmd} ${module_name}
                done
            done
        fi
    done
done
