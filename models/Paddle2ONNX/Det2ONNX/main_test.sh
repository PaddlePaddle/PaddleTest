# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env bash

cur_path=`pwd`

while getopts ":b:p:t:g:" opt
do
    case $opt in
        b)
        echo "test branch=$OPTARG"
        branch=$OPTARG
        ;;
        p)
        echo "py version=$OPTARG"
        py_cmd=$OPTARG
        ;;
        t)
        echo "repo=$OPTARG"
        repo=$OPTARG
        ;;
        g)
        echo "use gpu=$OPTARG"
        use_gpu=$OPTARG
        ;;
        ?)
        echo "未知参数"
        usage
    esac
done


use_key_get_value(){
    tmp=`grep $1 $2`
    key=${tmp%%::*}
    value=${tmp##*::}
    echo ${value}
}

main(){
    case $repo in
        (ce)
            ce
            ;;
        (det)
            excption=0
            rm -rf DetForONNX
            git clone -b ${branch} https://github.com/PaddlePaddle/PaddleDetection.git DetForONNX
            $py_cmd -m pip install -r DetForONNX/requirements.txt
            cd DetForONNX
            $py_cmd setup.py install
            cd -
            rm -rf DetForONNX/deploy/python/infer_for_onnx.py
            rm -rf DetForONNX/deploy/python/key_infer_for_onnx.py
            cp -r infer_for_onnx.py DetForONNX/deploy/python
            cp -r key_infer_for_onnx.py DetForONNX/deploy/python
            rm -rf DetForONNX/deploy/python/utils.py && cp -r utils_for_onnx.py DetForONNX/deploy/python/utils.py
            rm -rf models && rm -rf det_data && rm -rf log
            wget https://paddle-qa.bj.bcebos.com/Paddle2ONNX/data_set/det_data/det_data.tar && tar -xf det_data.tar && rm -rf det_data.tar

            for model_txt in ./models_txt/*
            do
                tmp=${model_txt%%.txt*}
                model_name=${tmp##*/}
                echo ${model_name} is comming!!!
                mkdir -p models/${model_name}/pretrain_model && mkdir -p models/${model_name}/infer_model && mkdir -p models/${model_name}/onnx_model
                mkdir -p models/${model_name}/test_data && mkdir -p models/${model_name}/input_np
                mkdir -p models/${model_name}/infer_output_np && mkdir -p models/${model_name}/onnx_output_np
                mkdir -p log/${model_name}

                yaml_path=`use_key_get_value yaml ${model_txt}`
                premodel_link=`use_key_get_value pretrain_link ${model_txt}`
                opt_ver=`use_key_get_value opt_ver ${model_txt}`
                atol=`use_key_get_value atol ${model_txt}`
                rtol=`use_key_get_value rtol ${model_txt}`
                diff_per=`use_key_get_value diff_per ${model_txt}`
                model_type=`use_key_get_value model_type ${model_txt}`
                #download pretrained model and test data
                wget -P models/${model_name}/pretrain_model ${premodel_link} > download_data.log 2>&1
                mv models/${model_name}/pretrain_model/*.pdparams models/${model_name}/pretrain_model/model.pdparams
                echo ++++++++++++++++++++++${model_name} from pretrained model export infer model!!!++++++++++++++++++++++
                $py_cmd DetForONNX/tools/export_model.py \
                     -c ${yaml_path} \
                     -o weights=models/${model_name}/pretrain_model/model.pdparams \
                     --output_dir=models/${model_name}/infer_model >> log/${model_name}/export.log 2>&1
                if [ $? -ne 0 ];then
                echo ++++++++++++++++++++++${model_name} export Failed!!!++++++++++++++++++++++
                echo ${model_name} export Failed!!! >> log/whole_test.log
                cat log/${model_name}/export.log
                excption=$(expr ${excption} + 1)
                continue
                else
                echo ++++++++++++++++++++++${model_name} export Success!!!++++++++++++++++++++++
                fi
                #from infer model export onnx model
                paddle2onnx \
                     --model_dir models/${model_name}/infer_model/${model_name} \
                     --model_filename model.pdmodel \
                     --params_filename model.pdiparams \
                     --opset_version ${opt_ver} \
                     --save_file models/${model_name}/onnx_model/model.onnx >> log/${model_name}/to_onnx.log 2>&1
                if [ $? -ne 0 ];then
                echo ++++++++++++++++++++++${model_name} to_onnx Failed!!!++++++++++++++++++++++
                echo ${model_name} to_onnx Failed!!! >> log/whole_test.log
                cat log/${model_name}/to_onnx.log
                excption=$(expr ${excption} + 1)
                continue
                else
                echo ++++++++++++++++++++++${model_name} to_onnx Success!!!++++++++++++++++++++++
                fi
                #push test_data into infer model and get output
                if [ ${use_gpu} == 'True' ]; then
                use_device=GPU
                else
                use_device=CPU
                fi
                if [ ${model_type} == 'KeyPoint' ]; then
                infer_script=key_infer_for_onnx.py
                else
                infer_script=infer_for_onnx.py
                fi
                $py_cmd DetForONNX/deploy/python/${infer_script} \
                     --model_dir=./models/${model_name}/infer_model/${model_name} \
                     --image_dir=./det_data \
                     --output_dir=./models/${model_name}/infer_output_image \
                     --input_np=./models/${model_name}/input_np \
                     --infer_output_np=./models/${model_name}/infer_output_np \
                     --device=${use_device} >> log/${model_name}/infer_predict.log 2>&1
                if [ $? -ne 0 ];then
                echo ++++++++++++++++++++++${model_name} infer predict Failed!!!++++++++++++++++++++++
                echo ${model_name} infer predict Failed!!! >> log/whole_test.log
                cat log/${model_name}/infer_predict.log
                excption=$(expr ${excption} + 1)
                continue
                else
                echo ++++++++++++++++++++++${model_name} infer predict Success!!!++++++++++++++++++++++
                fi
                #push test_data into onnx model and get output
                $py_cmd onnx.py \
                     --model_name models/${model_name} \
                     --use_gpu ${use_gpu} >> log/${model_name}/onnx_predict.log 2>&1
                if [ $? -ne 0 ];then
                echo ++++++++++++++++++++++${model_name} onnx predict Failed!!!++++++++++++++++++++++
                echo ${model_name} onnx predict Failed!!! >> log/whole_test.log
                cat log/${model_name}/onnx_predict.log
                excption=$(expr ${excption} + 1)
                continue
                else
                echo ++++++++++++++++++++++${model_name} onnx predict Success!!!++++++++++++++++++++++
                fi
                #compare paddle infer and onnx output
                $py_cmd compare_tool.py \
                     --model_name models/${model_name} \
                     --atol ${atol} \
                     --rtol ${rtol} >> log/${model_name}/final_compare.log 2>&1
                if [ $? -ne 0 ];then
                echo ++++++++++++++++++++++${model_name} finally acc compare Failed!!!++++++++++++++++++++++
                echo ${model_name} finally acc compare Failed!!! >> log/whole_test.log
                cat log/${model_name}/final_compare.log
                excption=$(expr ${excption} + 1)
                continue
                else
                echo ++++++++++++++++++++++${model_name} finally acc compare Success!!!++++++++++++++++++++++
                echo ${model_name} finally acc compare Success!!! >> log/whole_test.log
                fi
            done
            #return ${excption}
            #exit_code=`det`
            #all_error=`echo $?`
            echo "================================== final-results =================================="
            all_error=${excption}
            #echo ${exit_code}
            cat log/whole_test.log
            exit ${all_error}
            ;;
        (*)
            echo "Error command"
            usage
            ;;
    esac
}

main
