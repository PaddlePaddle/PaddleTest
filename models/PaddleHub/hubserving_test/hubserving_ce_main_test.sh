#!/usr/bin/env bash

unset GREP_OPTIONS
cur_path=`pwd`

while getopts ":P:b:p:t:g:L:h:H:" opt
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
        g)
        echo "use gpu=$OPTARG"
        use_gpu=$OPTARG
        ;;
        L)
        echo "Linux sys=$OPTARG"
        linux_sys=$OPTARG
        ;;
        h)
        hub_config=$OPTARG
        ;;
        H)
        http_proxy=$OPTARG
        ;;
        ?)
        echo "未知参数"
        usage
    esac
done


build_env(){
    root_path=`pwd`
    $py_cmd -m pip install --upgrade pip
    $py_cmd -m pip install pytest
    $py_cmd -m pip install ${paddle}

    rm -rf Hub_all_module_ce_test
    git clone -b ${branch} https://github.com/PaddlePaddle/PaddleHub.git Hub_all_module_ce_test
    cd Hub_all_module_ce_test
    $py_cmd -m pip install -r requirements.txt
    $py_cmd setup.py install
    cd -

    hub config server==${hub_config}

#    mkdir hub
#    export HUB_HOME=hub

#    $py_cmd -m pip install Cython
#    $py_cmd -m pip install imageio
#    $py_cmd -m pip install imageio-ffmpeg

    if [[ ${linux_sys} = "ubuntu" ]]; then
    apt-get install libsndfile1 -y
    fi
    if [[ ${linux_sys} = "centos" ]]; then
    yum install -y libsndfile
    fi

    $py_cmd -m pip install librosa

    $py_cmd -m pip install sentencepiece
    $py_cmd -m pip install pypinyin --upgrade
    $py_cmd -m pip install paddlex==1.3.11
#
#    $py_cmd -m pip install ruamel.yaml
#    git clone -b release/v0.1 https://github.com/PaddlePaddle/Parakeet && cd Parakeet && $py_cmd -m pip install -e .
#    $py_cmd -m pip install librosa

#    git clone https://github.com/PaddlePaddle/DeepSpeech.git && cd DeepSpeech && git reset --hard b53171694e7b87abe7ea96870b2f4d8e0e2b1485 && cd deepspeech/decoders/ctcdecoder/swig && sh setup.sh
#    cd ${root_path}
}

main(){
    case $repo in
        (build_env)
            build_env
            ;;
        (hubserving_test)

#            build_env

            serving_excption=0
            serving_success=0
            serving_fail_list=

            predict_excption=0
            predict_success=0
            predict_fail_list=
            run_time=`date +"%Y-%m-%d_%H:%M:%S"`

            rm -rf all_module_log
            mkdir all_module_log
            mkdir -p all_module_log/serving && mkdir -p all_module_log/predict

            echo "======================> run time is ${run_time} "

            echo "======================> paddle version commit: "
            $py_cmd -c "import paddle; print('paddle version:',paddle.__version__,'\npaddle commit:',paddle.version.commit)";

            echo "======================> python version: "
            $py_cmd -c 'import sys; print(sys.version_info[:])'

            wget -q --no-proxy https://paddle-qa.bj.bcebos.com/PaddleHub/data/hub_data.tar
            tar -xzf hub_data.tar && mv hub_data/* . && rm -rf hub_data.tar

            wget -q --no-proxy https://paddle-qa.bj.bcebos.com/PaddleHub/hubserving_test/hubserving_py.tar
            tar -xzf hubserving_py.tar && rm -rf hubserving_py.tar

            for hub_module in `cat hubserving_all.txt`
            do

            # 修改为CPU预测
            if [[ ${use_gpu} = "False" ]]; then
            sed -i "s/paddle.set_device('gpu')/paddle.set_device('cpu')/g" hubserving_py/test_${hub_module}.py
            sed -i "s/use_gpu=True/use_gpu=False/g" hubserving_py/test_${hub_module}.py
            fi

            echo ++++++++++++++++++++++ ${hub_module} start installing !!!++++++++++++++++++++++
            export https_proxy=${http_proxy}
            export http_proxy=${http_proxy}
            export no_proxy=bcebos.com
            hub install ${hub_module}

            echo ++++++++++++++++++++++ ${hub_module} start serving !!!++++++++++++++++++++++
            unset https_proxy
            unset http_proxy
            nohup hub serving start -m ${hub_module} 2>&1 & # >> all_module_log/serving/${hub_module}.log

            if [ $? -ne 0 ];then
                echo ++++++++++++++++++++++ ${hub_module} serving Failed!!!++++++++++++++++++++++
                serving_excption=$(expr ${serving_excption} + 1)
                serving_fail_list="${serving_fail_list} ${hub_module}"
                cat all_module_log/serving/${hub_module}.log
                else
                echo ++++++++++++++++++++++ ${hub_module} serving Success!!!++++++++++++++++++++++
                serving_success=$(expr ${serving_success} + 1)
            fi

            sleep 10

            echo ++++++++++++++++++++++ ${hub_module} start predicting !!!++++++++++++++++++++++
            $py_cmd hubserving_py/test_${hub_module}.py # >> all_module_log/predict/${hub_module}.log

            if [ $? -ne 0 ];then
                echo ++++++++++++++++++++++ ${hub_module} predict Failed!!!++++++++++++++++++++++
                predict_excption=$(expr ${predict_excption} + 1)
                predict_fail_list="${predict_fail_list} ${hub_module}"
                cat all_module_log/predict/${hub_module}.log
                else
                echo ++++++++++++++++++++++ ${hub_module} predict Success!!!++++++++++++++++++++++
                predict_success=$(expr ${predict_success} + 1)
            fi

            sleep 2

            echo ++++++++++++++++++++++ ${hub_module} stop serving !!!++++++++++++++++++++++
            hub serving stop -m ${hub_module}

#            rm -rf hub/.paddlehub/modules
#            rm -rf hub/.paddlehub/tmp
#
#            rm -rf hub/.paddlenlp/*
#            rm -rf hub/.paddleocr/*
#            rm -rf hub/.paddleseg/*
#            rm -rf hub/.paddlespeech/*

            rm -rf /root/.paddlehub/modules
            rm -rf /root/.paddlehub/tmp

            rm -rf /root/.paddlenlp/*
            rm -rf /root/.paddleocr/*
            rm -rf /root/.paddleseg/*
            rm -rf /root/.paddlespeech/*

#            sleep 5
            done

            echo "================================== final-results =================================="
#            if [[ -e "log/whole_fail.log" ]]; then
#            cat log/whole_fail.log
#            fi

            echo "serving_success = ${serving_success}"
            echo "serving_excption = ${serving_excption}"

            echo "predict_success = ${predict_success}"
            echo "predict_excption = ${predict_excption}"

            echo "serving_fail_list is: ${serving_fail_list}"
            echo "predict_fail_list is: ${predict_fail_list}"

            error_code=$(expr ${serving_excption} + ${predict_excption})

            exit ${error_code}

        (*)
            echo "Error command"
            usage
            ;;
    esac
}

main
