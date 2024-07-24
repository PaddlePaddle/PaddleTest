#!/bin/bash

set -e

MEM_SIZE=16

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
    last_status=${PIPESTATUS[0]}
    if [[ $last_status != 0 ]];then
        exit 1
    fi
}


#################################################### test_model ######################################################
# 获取python的绝对路径
PYTHONPATH="python"
# 获取当前脚本的绝对路径，获得基准目录
BASE_PATH=$(cd "$(dirname $0)"; pwd)
MODULE_OUTPUT_PATH=${BASE_PATH}/outputs
# 安装paddlex，完成环境准备
install_deps_cmd="pip install -e . && paddlex --install -y"
eval ${install_deps_cmd}

declare -A weight_dict
declare -A model_dict

IFS='*'
modules_info_file=${BASE_PATH}/run_suite.txt
modules_info_list=($(cat ${modules_info_file}))

for modules_info in ${modules_info_list[@]}; do
    IFS='='
    model_list=''
    info_list=($modules_info)
    for module_info in ${info_list[@]}; do
        IFS=$'\n'
        if [[ $module_info == *check_dataset_yaml* ]]; then
            # 数据准备，获取模型信息和运行模式
            lines=(${module_info})
            line_num=0
            module_name=$(func_parser_value "${lines[line_num]}")
            line_num=`expr $line_num + 1`
            check_dataset_yaml=$(func_parser_value "${lines[line_num]}")
            line_num=`expr $line_num + 1`
            dataset_url=https:$(func_parser_dataset_url "${lines[line_num]}")
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
            download_dataset_cmd="${PYTHONPATH} ${BASE_PATH}/checker.py --download_dataset --config_path ${check_dataset_yaml} --dataset_url ${dataset_url}"
            run_command ${download_dataset_cmd} ${module_name}
            model_output_path=${MODULE_OUTPUT_PATH}/${module_name}_dataset_check
            check_dataset_cmd="${PYTHONPATH} main.py -c ${check_dataset_yaml} -o Global.mode=check_dataset -o Global.output=${model_output_path} "
            run_command ${check_dataset_cmd} ${module_name}
            checker_cmd="${PYTHONPATH} ${BASE_PATH}/checker.py --check --check_dataset_result --output ${model_output_path} --module_name ${module_name}"
            run_command ${checker_cmd} ${module_name}

        elif [[ ! -z $module_info ]]; then
            for config_path in $module_info;do
                config_path=$(func_parser_value "${config_path}")
                batch_size=`cat $config_path | grep  -m 1 batch_size | awk  {'print$NF'}`
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
                yaml_name=${config_path##*/}
                model_name=${yaml_name%.*}
                model_list="${model_list} ${model_name}"
                model_output_path=${MODULE_OUTPUT_PATH}/${module_name}_output/${model_name}
                evaluate_weight_path=${model_output_path}/${best_weight_path}
                inference_weight_path=${model_output_path}/${inference_model_dir}
                weight_dict[$model_name]="$inference_weight_path"
                mkdir -p $model_output_path
                IFS=$'|'
                run_model_list=(${run_model})
                for mode in ${run_model_list[@]};do
                    # 根据config运行各模型的train和evaluate
                    run_mode_cmd="${PYTHONPATH} main.py -c ${config_path} -o Global.mode=${mode} -o Global.output=${model_output_path} -o Train.epochs_iters=${epochs_iters} -o Train.batch_size=${batch_size} -o Evaluate.weight_path=${evaluate_weight_path} -o Predict.model_dir=${inference_weight_path}"
                    run_command ${run_mode_cmd} ${module_name}
                done
                check_options_list=(${check_options})
                for check_option in ${check_options_list[@]};do
                    # 运行产出检查脚本
                    checker_cmd="${PYTHONPATH} ${BASE_PATH}/checker.py --check --$check_option --output ${model_output_path} --check_weights_items ${check_weights_items} --module_name ${module_name}"
                    run_command ${checker_cmd} ${module_name}
                done
            done
            model_dict[$module_name]="$model_list"
        fi
    done
done

#################################################### test_pipeline ######################################################
PIPELINE='image_classification instance_segmentation object_detection OCR semantic_segmentation'
DEMO_IMG='general_image_classification_001.jpg general_instance_segmentation_004.png general_object_detection_002.png general_ocr_002.png  general_semantic_segmentation_002.png'
BASE_URL='https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/'

IFS=$' '
PIPLINE_LIST=($PIPELINE)
DEMO_IMG_LIST=($DEMO_IMG)

length=${#PIPLINE_LIST[@]}

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
	cmd="paddlex --pipeline ${pipeline} --model ${model} --model_dir ${model_dir} --input ${img} --device gpu:0"
	eval $cmd
    last_status=${PIPESTATUS[0]}
    if [[ $last_status != 0 ]];then
        exit 1
    fi
	cd -
}

for (( i=0; i<$length; i++ ));do
	pipeline_name=${PIPLINE_LIST[$i]}
	image="${BASE_URL}${DEMO_IMG_LIST[$i]}"
	models=${model_dict[$pipeline_name]}
	if [[ $pipeline_name == OCR ]];then
        IFS=' '
        ocr_det_model=(${model_dict["text_detection"]})
        ocr_rec_model=(${model_dict["text_recognition"]})
		for det_model in ${ocr_det_model[@]};do
			for rec_model in ${ocr_rec_model[@]};do
				model_name="$det_model $rec_model"
                det_model_dir=${weight_dict[$det_model]}
                rec_model_dir=${weight_dict[$rec_model]}
                model_dir="$det_model_dir $rec_model_dir"
				check_pipeline $pipeline_name "$model_name" "$model_dir" $image
			done
		done
	else
		IFS=' '
		for model in $models;do
            model_dir=${weight_dict[$model]}
			check_pipeline $pipeline_name $model $model_dir $image
		done
	fi
done
