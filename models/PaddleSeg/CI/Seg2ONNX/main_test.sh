# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
        (seg)
            excption=0
#            rm -rf SegForONNX
#            git clone -b ${branch} https://github.com/PaddlePaddle/PaddleSeg.git SegForONNX
#            $py_cmd -m pip install -r SegForONNX/requirements.txt
#            cd SegForONNX
#            $py_cmd setup.py install
#            cd -
            rm -rf models && rm -rf seg_data && rm -rf onnx_log
#            wget https://paddle-qa.bj.bcebos.com/Paddle2ONNX/data_set/seg_data/seg_data.tar && tar -xf seg_data.tar && rm -rf seg_data.tar
            #cd SegForONNX
            argmax_situation="with_argmax"
            for model_txt in ./models_txt/*
            do
            for argmax_opt in ${argmax_situation}
            do
                tmp=${model_txt%%.txt*}
                model_name=${tmp##*/}
                echo ${model_name} ${argmax_opt} is comming!!!
                mkdir -p models/${model_name}/${argmax_opt}/pretrain_model && mkdir -p models/${model_name}/${argmax_opt}/infer_model && mkdir -p models/${model_name}/${argmax_opt}/onnx_model
                mkdir -p models/${model_name}/${argmax_opt}/test_data && mkdir -p models/${model_name}/${argmax_opt}/input_np
                mkdir -p models/${model_name}/${argmax_opt}/infer_output_np && mkdir -p models/${model_name}/${argmax_opt}/onnx_output_np
                mkdir -p onnx_log/${model_name}/${argmax_opt}

                yaml_path=`use_key_get_value yaml ${model_txt}`
                premodel_link=`use_key_get_value pretrain_link ${model_txt}`
                opt_ver=`use_key_get_value opt_ver ${model_txt}`
                atol=`use_key_get_value atol ${model_txt}`
                rtol=`use_key_get_value rtol ${model_txt}`
                diff_per=`use_key_get_value diff_per ${model_txt}`
                if [ ${argmax_opt} == 'with_argmax' ]; then
                without_argmax=False
                else
                without_argmax=True
                fi
                #download pretrained model and test data
                wget -P models/${model_name}/${argmax_opt}/pretrain_model ${premodel_link} > download_data.log 2>&1
                echo ++++++++++++++++++++++${model_name} ${argmax_opt} from pretrained model export infer model!!!++++++++++++++++++++++
                if [ ${argmax_opt} == 'with_argmax' ]; then
                $py_cmd export.py \
                     --config ${yaml_path} \
                     --model_path models/${model_name}/${argmax_opt}/pretrain_model/model.pdparams \
                     --save_dir models/${model_name}/${argmax_opt}/infer_model >> onnx_log/${model_name}/${argmax_opt}/export.log 2>&1
                else
                $py_cmd export.py \
                     --config ${yaml_path} \
                     --model_path models/${model_name}/${argmax_opt}/pretrain_model/model.pdparams \
                     --save_dir models/${model_name}/${argmax_opt}/infer_model \
                     --without_argmax >> onnx_log/${model_name}/${argmax_opt}/export.log 2>&1
                fi
                if [ $? -ne 0 ];then
                echo ++++++++++++++++++++++${model_name} ${argmax_opt} export Failed!!!++++++++++++++++++++++
                echo ${model_name}/${argmax_opt} export Failed!!! >> onnx_log/whole_test.log
                cat onnx_log/${model_name}/${argmax_opt}/export.log
                excption=$(expr ${excption} + 1)
                continue
                else
                echo ++++++++++++++++++++++${model_name} ${argmax_opt} export Success!!!++++++++++++++++++++++
                fi
                #from infer model export onnx model
                paddle2onnx \
                     --model_dir models/${model_name}/${argmax_opt}/infer_model \
                     --model_filename model.pdmodel \
                     --params_filename model.pdiparams \
                     --opset_version ${opt_ver} \
                     --save_file models/${model_name}/${argmax_opt}/onnx_model/model.onnx >> onnx_log/${model_name}/${argmax_opt}/to_onnx.log 2>&1
                if [ $? -ne 0 ];then
                echo ++++++++++++++++++++++${model_name} ${argmax_opt} to_onnx Failed!!!++++++++++++++++++++++
                echo ${model_name}/${argmax_opt} to_onnx Failed!!! >> onnx_log/whole_test.log
                cat onnx_log/${model_name}/${argmax_opt}/to_onnx.log
                excption=$(expr ${excption} + 1)
                continue
                else
                echo ++++++++++++++++++++++${model_name} ${argmax_opt} to_onnx Success!!!++++++++++++++++++++++
                echo ${model_name}/${argmax_opt} export to onnx Success!!! >> onnx_log/whole_test.log
                fi
#                #push test_data into infer model and get output
#                $py_cmd infer_for_onnx.py \
#                     --config models/${model_name}/${argmax_opt}/infer_model/*.yaml \
#                     --image_path seg_data \
#                     --save_dir models/${model_name}/${argmax_opt}/infer_output_image \
#                     --model_name models/${model_name} \
#                     --with_argmax ${argmax_opt} >> log/${model_name}/${argmax_opt}/infer_predict.log 2>&1
#                if [ $? -ne 0 ];then
#                echo ++++++++++++++++++++++${model_name} ${argmax_opt} infer predict Failed!!!++++++++++++++++++++++
#                echo ${model_name}/${argmax_opt} infer predict Failed!!! >> log/whole_test.log
#                cat log/${model_name}/${argmax_opt}/infer_predict.log
#                excption=$(expr ${excption} + 1)
#                continue
#                else
#                echo ++++++++++++++++++++++${model_name} ${argmax_opt} infer predict Success!!!++++++++++++++++++++++
#                fi
#                #push test_data into onnx model and get output
#                $py_cmd onnx.py \
#                     --model_name models/${model_name} \
#                     --use_gpu ${use_gpu} \
#                     --with_argmax ${argmax_opt} >> log/${model_name}/${argmax_opt}/onnx_predict.log 2>&1
#                if [ $? -ne 0 ];then
#                echo ++++++++++++++++++++++${model_name} ${argmax_opt} onnx predict Failed!!!++++++++++++++++++++++
#                echo ${model_name}/${argmax_opt} onnx predict Failed!!! >> log/whole_test.log
#                cat log/${model_name}/${argmax_opt}/onnx_predict.log
#                excption=$(expr ${excption} + 1)
#                continue
#                else
#                echo ++++++++++++++++++++++${model_name} ${argmax_opt} onnx predict Success!!!++++++++++++++++++++++
#                fi
#                #compare paddle infer and onnx output
#                $py_cmd compare_tool.py \
#                     --model_name models/${model_name} \
#                     --atol ${atol} \
#                     --rtol ${rtol} \
#                     --diff_per ${diff_per} \
#                     --with_argmax ${argmax_opt} >> log/${model_name}/${argmax_opt}/final_compare.log 2>&1
#                if [ $? -ne 0 ];then
#                echo ++++++++++++++++++++++${model_name} ${argmax_opt} finally acc compare Failed!!!++++++++++++++++++++++
#                echo ${model_name}/${argmax_opt} finally acc compare Failed!!! >> log/whole_test.log
#                cat log/${model_name}/${argmax_opt}/final_compare.log
#                excption=$(expr ${excption} + 1)
#                continue
#                else
#                echo ++++++++++++++++++++++${model_name} ${argmax_opt} finally acc compare Success!!!++++++++++++++++++++++
#                echo ${model_name}/${argmax_opt} finally acc compare Success!!! >> log/whole_test.log
#                fi
            done
            done
            #return ${excption}
            #exit_code=`seg`
            #all_error=`echo $?`
            echo "================================== final-results =================================="
            all_error=${excption}
            #echo ${exit_code}
            cat onnx_log/whole_test.log
            exit ${all_error}
            ;;
        (*)
            echo "Error command"
            usage
            ;;
    esac
}

main
