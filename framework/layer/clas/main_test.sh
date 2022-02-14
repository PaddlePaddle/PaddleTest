#!/usr/bin/env bash

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

unset GREP_OPTIONS
cur_path=`pwd`

while getopts ":P:b:p:t:u:" opt
do
    case $opt in
        P)
        echo "test paddle=$OPTARG"
        paddle=$OPTARG
        ;;
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
        u)
        echo "upload=$OPTARG"
        upload=$OPTARG
        ;;
        ?)
        echo "未知参数"
        usage
    esac
done

build_env(){
    $py_cmd -m pip install --upgrade pip
    $py_cmd -m pip install pytest
    $py_cmd -m pip install ${paddle}

}


build_clas(){

    build_env

    #download Layer_CE_code from bos. These code is from PaddleClas and has been fixed by paddle-qa for layer test
    wget -q --no-proxy https://paddle-qa.bj.bcebos.com/PaddleLayerTest/paddleclas/Layer_CE_code.tar
    tar -xzf Layer_CE_code.tar && mv Layer_CE_code/* . && rm -rf Layer_CE_code.tar && rm -rf Layer_CE_code

    rm -rf PaddleClasLayerTest
    git clone -b ${branch} https://github.com/PaddlePaddle/PaddleClas.git PaddleClasLayerTest
    $py_cmd -m pip install -r PaddleClasLayerTest/requirements.txt
    rm -rf PaddleClasLayerTest/ppcls/engine/train/train.py
    cp -r Layer_CE_train.py PaddleClasLayerTest/ppcls/engine/train
    mv PaddleClasLayerTest/ppcls/engine/train/Layer_CE_train.py PaddleClasLayerTest/ppcls/engine/train/train.py
    rm -rf PaddleClasLayerTest/ppcls/engine/evaluation/classification.py
    cp -r Layer_CE_classification.py PaddleClasLayerTest/ppcls/engine/evaluation
    mv PaddleClasLayerTest/ppcls/engine/evaluation/Layer_CE_classification.py PaddleClasLayerTest/ppcls/engine/evaluation/classification.py
    rm -rf PaddleClasLayerTest/ppcls/engine/engine.py
    cp -r Layer_CE_engine.py PaddleClasLayerTest/ppcls/engine
    mv PaddleClasLayerTest/ppcls/engine/Layer_CE_engine.py PaddleClasLayerTest/ppcls/engine/engine.py

    \cp -f paddleclas_model_py/* PaddleClasLayerTest/ppcls/arch/backbone/legendary_models

    wget -q --no-proxy https://paddle-qa.bj.bcebos.com/PaddleLayerTest/paddleclas/ILSVRC2012.tar
    tar -xf ILSVRC2012.tar -C PaddleClasLayerTest/dataset && rm -rf ILSVRC2012.tar

}

main(){
    case $repo in
        (build_clas_case)
            train_excption=0
            train_fail_list=0
            export FLAGS_cudnn_deterministic=True

            build_clas
            rm -rf log && rm -rf output

            wget -q --no-proxy https://xly-devops.bj.bcebos.com/home/bos_new.tar.gz
            tar -xzf bos_new.tar.gz

            for model_yaml in `ls clas_upload_yaml`
            do
                model_arch=`grep -i "Arch" -A 1 clas_upload_yaml/${model_yaml} | grep "name" | awk -F': ' '{print $2}'`
                pdparams_output=`grep "output_dir:" clas_upload_yaml/${model_yaml} | awk -F': ' '{print $2}'`
                model_case=${model_yaml%.*}

                mkdir -p ${pdparams_output}/${model_arch}

                echo ++++++++++++++++++++++ ${model_case} start training !!!++++++++++++++++++++++
                $py_cmd PaddleClasLayerTest/tools/train.py -c clas_upload_yaml/${model_yaml}

                if [ $? -ne 0 ];then
                echo ++++++++++++++++++++++ ${model_case} train Failed!!!++++++++++++++++++++++
                train_excption=$(expr ${train_excption} + 1)
                train_fail_list="${train_fail_list} ${model_case}"
                continue
                else
                echo ++++++++++++++++++++++ ${model_case} train Success!!!++++++++++++++++++++++
                fi

                if [[ ${upload} = exp ]]; then
                #change backward and forward name
                mv ${pdparams_output}/${model_arch}/backward ${pdparams_output}/${model_arch}/backward_exp
                mv ${pdparams_output}/${model_arch}/forward ${pdparams_output}/${model_arch}/forward_exp
                mv ${pdparams_output}/${model_arch}/train.log ${pdparams_output}/${model_arch}/train_exp.log

                tar czvf ${model_case}.tar -C ${pdparams_output} ${model_arch}
                echo ++++++++++++++++++++++ ${model_case} start uploading to bos !!!++++++++++++++++++++++
                $py_cmd BosClient.py ${model_case}.tar paddle-qa/PaddleLayerTest/paddleclas/Linux/V100/py37 https://paddle-qa.bj.bcebos.com/PaddleLayerTest/paddleclas/Linux/V100/py37
                rm -rf ${model_case}.tar
                fi

            done
            echo "train_excption = ${train_excption}"
            echo "train_fail_list is: ${train_fail_list}"
            ;;
        (paddleclas)
            train_excption=0
            forward_excption=0
            backward_excption=0
            train_fail_list=
            run_time=`date +"%Y_%m_%d_%H_%M_%S"`
            export FLAGS_cudnn_deterministic=True

            build_clas

            echo "======================> run time is ${run_time} "

            echo "======================> paddle version commit: "
            $py_cmd -c "import paddle; print('paddle version:',paddle.__version__,'\npaddle commit:',paddle.version.commit)";

            echo "======================> python version: "
            python -c 'import sys; print(sys.version_info[:])'
#            git clone -b develop https://github.com/PaddlePaddle/PaddleTest.git
#            cp -r PaddleTest/framework/e2e/paddle_layer_test/clas/* .

            rm -rf log && rm -rf output

            for model_yaml in `ls paddleclas_yaml`
            do
                model_arch=`grep -i "Arch" -A 1 paddleclas_yaml/${model_yaml} | grep "name" | awk -F': ' '{print $2}'`
                pdparams_output=`grep "output_dir:" paddleclas_yaml/${model_yaml} | awk -F': ' '{print $2}'`
                model_case=${model_yaml%.*}
                mkdir -p log/${model_case}
                mkdir -p ${pdparams_output}/${model_arch}

                echo ++++++++++++++++++++++ ${model_case} start testing !!!++++++++++++++++++++++
                wget -q --no-proxy https://paddle-qa.bj.bcebos.com/PaddleLayerTest/${repo}/Linux/V100/py37/${model_case}.tar
                tar -xzf ${model_case}.tar -C ${pdparams_output} && rm -rf ${model_case}.tar

                echo "======================> ${model_case} conclusion: " >> log/whole_test.log

                echo ++++++++++++++++++++++ ${model_case} start training !!!++++++++++++++++++++++
                $py_cmd PaddleClasLayerTest/tools/train.py -c paddleclas_yaml/${model_yaml}

                if [ $? -ne 0 ];then
                echo ++++++++++++++++++++++ ${model_case} train Failed!!!++++++++++++++++++++++
                train_excption=$(expr ${train_excption} + 1)
                train_fail_list="${train_fail_list} ${model_case}"
                continue
                else
                echo ++++++++++++++++++++++ ${model_case} train Success!!!++++++++++++++++++++++
                fi

                cp -r ${pdparams_output}/${model_arch}/train.log log/${model_case}

                echo ++++++++++++++++++++++ ${model_case} start comparing forward !!!++++++++++++++++++++++
                $py_cmd pdparams_compare_tool.py \
                    --params_exp ${pdparams_output}/${model_arch}/forward_exp \
                    --params_res ${pdparams_output}/${model_arch}/forward >> log/${model_case}/forward.log 2>&1

                fail_count=`grep "failed file number:" log/${model_case}/forward.log | awk -F':  ' '{print $2}'`
                succ_count=`grep "Success file number:" log/${model_case}/forward.log | awk -F':  ' '{print $2}'`
                fail_list=`grep "failed file list:" log/${model_case}/forward.log | awk -F':  ' '{print $2}'`
                succ_list=`grep "Success file list:" log/${model_case}/forward.log | awk -F':  ' '{print $2}'`

                if [[ ${fail_count} != "0" ]] || [[ "${fail_count}" = "" ]];then
                echo ++++++++++++++++++++++ ${model_case} forward testing Failed!!!++++++++++++++++++++++
                forward_excption=$(expr ${forward_excption} + 1)
                echo "==============> ${model_case} forward Failed !!!" >> log/whole_fail.log
                echo "fail_count is: ${fail_count}" >> log/whole_fail.log
                echo "fail_list is: ${fail_list}" >> log/whole_fail.log

                echo "==============> ${model_case} forward Failed !!!" >> log/whole_test.log
                echo "succ_count is: ${succ_count}" >> log/whole_test.log
                echo "succ_list is: ${succ_list}" >> log/whole_test.log
                echo "fail_count is: ${fail_count}" >> log/whole_test.log
                echo "fail_list is: ${fail_list}" >> log/whole_test.log
                else
                echo ++++++++++++++++++++++ ${model_case} forward testing Success!!!++++++++++++++++++++++
                echo "==============> ${model_case} forward Success !!!" >> log/whole_test.log
                echo "succ_count is: ${succ_count}" >> log/whole_test.log
                echo "succ_list is: ${succ_list}" >> log/whole_test.log
                echo "fail_count is: ${fail_count}" >> log/whole_test.log
                echo "fail_list is: ${fail_list}" >> log/whole_test.log
                fi

                echo ++++++++++++++++++++++ ${model_case} start comparing backward !!!++++++++++++++++++++++
                $py_cmd pdparams_compare_tool.py \
                    --params_exp ${pdparams_output}/${model_arch}/backward_exp \
                    --params_res ${pdparams_output}/${model_arch}/backward >> log/${model_case}/backward.log 2>&1

                fail_count=`grep "failed file number:" log/${model_case}/backward.log | awk -F':  ' '{print $2}'`
                succ_count=`grep "Success file number:" log/${model_case}/backward.log | awk -F':  ' '{print $2}'`
                fail_list=`grep "failed file list:" log/${model_case}/backward.log | awk -F':  ' '{print $2}'`
                succ_list=`grep "Success file list:" log/${model_case}/backward.log | awk -F':  ' '{print $2}'`

                if [[ ${fail_count} != "0" ]] || [[ "${fail_count}" = "" ]];then
                echo ++++++++++++++++++++++ ${model_case} backward testing Failed!!!++++++++++++++++++++++
                backward_excption=$(expr ${backward_excption} + 1)
                echo "==============> ${model_case} backward Failed !!!" >> log/whole_fail.log
                echo "fail_count is: ${fail_count}" >> log/whole_fail.log
                echo "fail_list is: ${fail_list}" >> log/whole_fail.log

                echo "==============> ${model_case} backward Failed !!!" >> log/whole_test.log
                echo "succ_count is: ${succ_count}" >> log/whole_test.log
                echo "succ_list is: ${succ_list}" >> log/whole_test.log
                echo "fail_count is: ${fail_count}" >> log/whole_test.log
                echo "fail_list is: ${fail_list}" >> log/whole_test.log
                else
                echo ++++++++++++++++++++++ ${model_case} backward testing Success!!!++++++++++++++++++++++
                echo "==============> ${model_case} backward Success !!!" >> log/whole_test.log
                echo "succ_count is: ${succ_count}" >> log/whole_test.log
                echo "succ_list is: ${succ_list}" >> log/whole_test.log
                echo "fail_count is: ${fail_count}" >> log/whole_test.log
                echo "fail_list is: ${fail_list}" >> log/whole_test.log
                fi

#                rm -rf ${pdparams_output}/${model_arch}
            done

            error_code=$(expr ${train_excption} + ${forward_excption} + ${backward_excption})

            echo ++++++++++++++++++++++ upload ${run_time}.log to bos !!!++++++++++++++++++++++
            tar -czf clas_layer_${run_time}.tar log
            wget -q --no-proxy https://xly-devops.bj.bcebos.com/home/bos_new.tar.gz
            tar -xzf bos_new.tar.gz
            $py_cmd BosClient.py clas_layer_${run_time}.tar paddle-qa/PaddleLayerTest/log/paddleclas/Linux/V100/py37 https://paddle-qa.bj.bcebos.com/PaddleLayerTest/log/paddleclas/Linux/V100/py37

            if [[ ${upload} = res ]] && [[ ${error_code} != 0 ]]; then
            echo "backward_excption = ${backward_excption}"
            tar -czf output_${run_time}.tar output
            fi

            echo "================================== final-results =================================="
            cat log/whole_fail.log
            echo "train_excption = ${train_excption}"
            echo "train_fail_list is: ${train_fail_list}"
            echo "forward_excption = ${forward_excption}"
            echo "backward_excption = ${backward_excption}"

            exit ${error_code}
            ;;
        (*)
            echo "Error command"
            usage
            ;;
    esac
}

main
