#!/bin/bash

SUITE=$1
NEW_MODEL_INFO_FILE=$2
MEM_SIZE=16


set -x
if [[ $SUITE != 'PaddleX' ]];then
    set -e
else
    failed_model_info=""
fi




#################################################### Functions ######################################################

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

# 运行命令并输出结果，PR级CI失败会重跑3次并异常退出，增量级和全量级会记录失败命令，最后打印失败的命令并异常退出
function run_command(){
    set +e
    command=$1
    module_name=$2
    time_stamp=$(date +"%Y-%m-%d %H:%M:%S")
    command="timeout 30m ${command}"
    printf "\e[32m|%-20s| %-20s | %-50s | %-20s\n\e[0m" "[${time_stamp}]" ${module_name} "Run ${command}"
    eval $command
    last_status=${PIPESTATUS[0]}
    if [[ $SUITE != 'PaddleX' ]];then
        n=1
        # Try 3 times to run command if it fails
        while [[ $last_status != 0 ]]; do
            sleep 10
            n=`expr $n + 1`
            printf "\e[32m|%-20s| %-20s | %-50s | %-20s\n\e[0m" "[${time_stamp}]" ${module_name} "Retrying $n times with command: ${command}"
            eval $command
            last_status=${PIPESTATUS[0]}
            if [[ $n -eq 3 && $last_status != 0 ]]; then
                echo "Retry 3 times failed with command: ${command}"
                exit 1
            fi
        done
    else
        if [[ $last_status != 0 ]];then
            failed_model_info="$failed_model_info \n ${module_name} | command: ${command}"
            echo "Run ${command} failed"
        fi
    fi
    set -e
}

# 准备数据集并做数据校验
function prepare_dataset(){
    if [[ $dataset_url == 'https:' ]];then
        train_data_file=""
        return
    fi
    download_dataset_cmd="${PYTHON_PATH} ${BASE_PATH}/checker.py --download_dataset --config_path ${check_dataset_yaml} --dataset_url ${dataset_url}"
    run_command ${download_dataset_cmd} ${module_name}
    model_output_path=${MODULE_OUTPUT_PATH}/${module_name}_dataset_check
    check_dataset_cmd="${PYTHON_PATH} main.py -c ${check_dataset_yaml} -o Global.mode=check_dataset -o Global.output=${model_output_path} "
    run_command ${check_dataset_cmd} ${module_name}
    checker_cmd="${PYTHON_PATH} ${BASE_PATH}/checker.py --check --check_dataset_result --output ${model_output_path} --module_name ${module_name}"
    run_command ${checker_cmd} ${module_name}
    dataset_dir=`cat $check_dataset_yaml | grep  -m 1 dataset_dir | awk  {'print$NF'}| sed 's/"//g'`
    if [[ ! -z $train_list_name ]]; then
        train_data_file=${dataset_dir}/${train_list_name}
        mv $train_data_file $train_data_file.bak
    fi
}

# 对给定的模型列表运行模型相应的train和evaluate等操作
function run_models(){
    config_files=$1
    for config_path in $config_files;do
        config_path=$(func_parser_value "${config_path}")
        batch_size=`cat $config_path | grep  -m 1 batch_size | awk  {'print$NF'}`
        device=`cat $config_path | grep  -m 1 device | awk  {'print$NF'}`
        IFS=$','
        device_list=(${device})
        device_num=${#device_list[@]}
        IFS=$' '
        # 根据内存大小调整batch_size
        if [[ $MEM_SIZE -lt 16 ]];then
            if [[ $batch_size -ge 4 ]];then
                batch_size=`expr $batch_size / 4`
            else
                batch_size=1
            fi
        elif [[ $MEM_SIZE -lt 32 ]];then
            if [[ $batch_size -ge 2 ]];then
                batch_size=`expr $batch_size / 2`
            else
                batch_size=1
            fi
        fi
        # 根据batch_size和device_num调整数据集的数量
        data_num=`expr $device_num \* $batch_size`
        if [[ ! -z $train_data_file ]]; then
            if [[ $module_name == ts* ]]; then
                data_num=`expr $device_num \* $batch_size \* 30`
                data_num=`expr $data_num + 1`
            fi
            head -n $data_num $train_data_file.bak > $train_data_file
        fi
        yaml_name=${config_path##*/}
        model_name=${yaml_name%.*}
        model_list="${model_list} ${model_name}"
        model_output_path=${MODULE_OUTPUT_PATH}/${module_name}_output/${model_name}
        evaluate_weight_path=${model_output_path}/${best_weight_path}
        if [[ $inference_model_dir == 'null' ]];then
            inference_weight_path="None"
        else
            inference_weight_path=${model_output_path}/${inference_model_dir}
        fi
        mkdir -p $model_output_path
        IFS=$'|'
        run_model_list=(${run_model})
        for mode in ${run_model_list[@]};do
            set +e
            black_model=`eval echo '$'"${mode}_black_list"|grep "^${model_name}$"`
            set -e
            if [[ ! -z $black_model ]];then
                # 黑名单模型，不运行
                echo "$model_name is in ${mode}_black_list, so skip it."
                continue
            fi
            # 适配导出模型时，需要指定输出路径
            base_mode_cmd="${PYTHON_PATH} main.py -c ${config_path} -o Global.mode=${mode} -o Train.epochs_iters=${epochs_iters} -o Train.batch_size=${batch_size} -o Evaluate.weight_path=${evaluate_weight_path} -o Predict.model_dir=${inference_weight_path}"
            if [[ $mode == "export" ]];then
                model_export_output_path=${model_output_path}/export
                mkdir -p $model_export_output_path
                weight_dict[$model_name]="$model_export_output_path"
                run_mode_cmd="${base_mode_cmd} -o Global.output=${model_export_output_path}"
            else
                run_mode_cmd="${base_mode_cmd} -o Global.output=${model_output_path}"
            fi
            run_command ${run_mode_cmd} ${module_name}
            # 在predict模式下，再次验证官方模型预测
            if [[ $mode == "predict" ]];then
                offcial_model_predict_cmd="${PYTHON_PATH} main.py -c ${config_path} -o Global.mode=${mode} -o Predict.model_dir=None -o Global.output=${model_output_path}_offical_predict"
                run_command ${offcial_model_predict_cmd} ${module_name}
            fi
        done
        if [[ ! -z $black_model ]];then
            # 黑名单模型，不做检查
            echo "$model_name is in ${mode}_black_list, so skip it."
            continue
        fi
        if [[ $check_options == 'null' ]]; then
            continue
        fi
        check_options_list=(${check_options})
        for check_option in ${check_options_list[@]};do
            # 运行产出检查脚本
            checker_cmd="${PYTHON_PATH} ${BASE_PATH}/checker.py --check --$check_option --output ${model_output_path} --check_weights_items ${check_weights_items} --module_name ${module_name}"
            run_command ${checker_cmd} ${module_name}
        done
        rm -rf ${model_output_path}/*[0-9]*
    done
    model_dict[$module_name]="$model_list"
}

# 准备数据集和运行模型
function prepare_and_run(){
    config_file_list=$1
    echo $config_file_list
    prepare_dataset
    run_models "${config_file_list}"
}


#################################################### PaddleX CI ######################################################
# PaddleX CI 区分为 PR 级、套件级、全量级和增量级
# PR 级：运行 PaddleX_simplify_models.txt 中的重点模型列表
# 套件级：运行 PaddleX_simplify_models.txt 中指定套件的部分模型
# 全量级：根据 ci_info.txt 抓取 PaddleX 支持的所有模型并测试
# 增量级：根据 ci_info.txt 和 传入的 changed_yaml.txt 抓取新增的模型，对新增模型进行增量测试

# 指定 python
PYTHON_PATH="python"
# 获取当前脚本的绝对路径，获得基准目录
BASE_PATH=$(cd "$(dirname $0)"; pwd)
MODULE_OUTPUT_PATH=${BASE_PATH}/outputs
pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple
declare -A weight_dict
declare -A model_dict

# 安装paddlex，完成环境准备
if [[ -z $SUITE ]]; then
    install_deps_cmd="pip install -e . && paddlex --install -y"
    modules_info_file=${BASE_PATH}/PaddleX_simplify_models.txt
elif [[ $SUITE == "PaddleX" ]];then
    install_deps_cmd="pip install -e . && paddlex --install -y"
    modules_info_file=${BASE_PATH}/ci_info.txt
else
    install_deps_cmd="pip install -e . && paddlex --install --use_local_repos $SUITE"
    modules_info_file=${BASE_PATH}/PaddleX_simplify_models.txt
fi

eval ${install_deps_cmd}
pip freeze > all_packages.txt


#################################################### 模型级测试 ######################################################
IFS=$' '
black_list_file=${BASE_PATH}/black_list.txt
set +e
all_black_list=`cat ${black_list_file} | grep All: | awk -F : {'print$2'}`
train_black_list=`cat ${black_list_file} | grep Train: | awk -F : {'print$2'}`
train_black_list="$all_black_list
$train_black_list"
evaluate_black_list=`cat ${black_list_file} | grep Evaluate: | awk -F : {'print$2'}`
evaluate_black_list="$all_black_list
$evaluate_black_list"
predict_black_list=`cat ${black_list_file} | grep Predict: | awk -F : {'print$2'}`
predict_black_list="$all_black_list
$predict_black_list"
export_black_list=`cat ${black_list_file} | grep Export: | awk -F : {'print$2'}`
export_black_list="$all_black_list
$export_black_list"
pipeline_black_list=`cat ${black_list_file} | grep Pipeline: | awk -F : {'print$2'}`
echo "----------------------- Black list info ------------------------
##############train_black_list###############
$train_black_list
##############evaluate_black_list###############
$evaluate_black_list
##############predict_black_list###############
$predict_black_list
##############export_black_list###############
$export_black_list
##############pipeline_black_list###############
$pipeline_black_list
-----------------------------------------------------------------"
set -e

IFS='*'
modules_info_list=($(cat ${modules_info_file}))
all_module_names=`cat $modules_info_file | grep module_name | awk -F ':' {'print$2'}`

unset http_proxy https_proxy
IFS=$' '
for modules_info in ${modules_info_list[@]}; do
    IFS='='
    model_list=''
    info_list=($modules_info)
    for module_info in ${info_list[@]}; do
        if [[ $module_info == *check_dataset_yaml* ]]; then
            # 数据准备，获取模型信息和运行模式
            IFS=$'\n'
            lines=(${module_info})
            line_num=0
            suite_name=$(func_parser_value "${lines[line_num]}")
            line_num=`expr $line_num + 1`
            module_name=$(func_parser_value "${lines[line_num]}")
            line_num=`expr $line_num + 1`
            check_dataset_yaml=$(func_parser_value "${lines[line_num]}")
            line_num=`expr $line_num + 1`
            dataset_url=https:$(func_parser_dataset_url "${lines[line_num]}")
            line_num=`expr $line_num + 1`
            train_list_name=$(func_parser_value "${lines[line_num]}")
            line_num=`expr $line_num + 1`
            run_model=$(func_parser_value "${lines[line_num]}")
            line_num=`expr $line_num + 1`
            check_options=$(func_parser_value "${lines[line_num]}")
            line_num=`expr $line_num + 1`
            check_weights_items=$(func_parser_value "${lines[line_num]}")
            line_num=`expr $line_num + 1`
            best_weight_path=$(func_parser_value "${lines[line_num]}")
            line_num=`expr $line_num + 1`
            inference_model_dir=$(func_parser_value "${lines[line_num]}")
            line_num=`expr $line_num + 1`
            epochs_iters=$(func_parser_value "${lines[line_num]}")
            if [[ $SUITE == "PaddleX" ]];then
                if [[ ! -z $NEW_MODEL_INFO_FILE ]];then
                    new_model_info=`cat $NEW_MODEL_INFO_FILE`
                    new_model_module_names=`cat $NEW_MODEL_INFO_FILE | awk -F '/' {'print$3'} | sort -u`
                    for new_module_info in ${new_model_module_names[@]};do
                        set +e
                        module=`echo "${all_module_names[@]}" | grep $new_module_info`
                        set -e
                        if [[ -z $module ]];then
                            echo "new module: $new_module_info is unsupported! Please contact with the developer or add new module info in ci_info.txt!"
                            exit 1
                        fi
                        if [[ $new_module_info == $module_name ]];then
                            module_info=`cat $NEW_MODEL_INFO_FILE | grep $module_name | xargs -n1 -I {} echo config_path:{}`
                            echo $module_info
                            prepare_and_run "${module_info}"
                        fi
                    done
                else
                    module_info=`ls paddlex/configs/${module_name} | xargs -n1 -I {} echo config_path:paddlex/configs/${module_name}/{}`
                    prepare_and_run "${module_info}"
                fi
                continue
            fi
        elif [[ ! -z $module_info ]]; then
            if [[ ! -z $SUITE && $SUITE == $suite_name ]];then
                prepare_and_run "${module_info}"
            elif [[ -z $SUITE ]];then
                prepare_and_run "${module_info}"
            else
                continue
            fi
        fi
    done
done

#################################################### 产线级测试 ######################################################
IFS=$'\n'
PIPELINE_YAML_LIST=`ls paddlex/pipelines | grep .yaml`

function check_pipeline() {
	pipeline=$1
	model=$2
    model_dir=$3
	img=$4
    output_dir_name=`echo $model | sed 's/ /_/g'`
    output_path=${MODULE_OUTPUT_PATH}/pipeline_output/${output_dir_name}
	rm -rf $output_path
	mkdir -p $output_path
	cd $output_path
    cmd="timeout 30m paddlex --pipeline ${pipeline} --input ${img}"
	eval $cmd
    last_status=${PIPESTATUS[0]}
    if [[ $last_status != 0 ]];then
        exit 1
    fi
	cd -
}

for pipeline_yaml in ${PIPELINE_YAML_LIST[@]};do
    pipeline_name=`cat paddlex/pipelines/${pipeline_yaml} | grep pipeline_name | awk {'print$2'}`
    set +e
    black_pipeline=`echo ${pipeline_black_list}|grep "^${pipeline_name}$"`
    set -e
    if [[ ! -z $black_pipeline ]];then
        # 黑名单产线，不运行
        echo "$pipeline_name is in pipeline_black_list, so skip it."
        continue
    fi
    input=`cat paddlex/pipelines/${pipeline_yaml} | grep input | awk {'print$2'}`
    check_pipeline $pipeline_name "" "" $input
done

if [[ $SUITE == 'PaddleX' && ! -z $failed_model_info ]];then
    echo $failed_model_info
    exit 1
fi
