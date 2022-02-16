#!/usr/bin/env bash

unset GREP_OPTIONS
cur_path=`pwd`

while getopts ":P:b:p:t:g:" opt
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
        ?)
        echo "未知参数"
        usage
    esac
done


build_env(){
    $py_cmd -m pip install --upgrade pip
    $py_cmd -m pip install pytest
    $py_cmd -m pip install ${paddle}

#    hub config server==${hub_config}

    rm -rf Hub_all_module_ce_test
    git clone -b ${branch} https://github.com/PaddlePaddle/PaddleHub.git Hub_all_module_ce_test
    cd Hub_all_module_ce_test
    $py_cmd -m pip install -r requirements.txt
    $py_cmd setup.py install
    cd -

#    $py_cmd -m pip install Cython
#    $py_cmd -m pip install imageio
#    $py_cmd -m pip install imageio-ffmpeg
#    apt-get install libsndfile1
#    $py_cmd -m pip install librosa
#
#    $py_cmd -m pip install ruamel.yaml
#    git clone -b release/v0.1 https://github.com/PaddlePaddle/Parakeet && cd Parakeet && $py_cmd -m pip install -e .
#    $py_cmd -m pip install librosa
}

main(){
    case $repo in
        (build_env)
            build_env
            ;;
        (hub_all_module_ce)

            build_env

            hub_excption=0
            hub_fail_list=
            run_time=`date +"%Y-%m-%d_%H:%M:%S"`

            rm -rf all_module_log
            mkdir all_module_log

            echo "======================> run time is ${run_time} "

            echo "======================> paddle version commit: "
            $py_cmd -c "import paddle; print('paddle version:',paddle.__version__,'\npaddle commit:',paddle.version.commit)";

            echo "======================> python version: "
            $py_cmd -c 'import sys; print(sys.version_info[:])'

            wget -q --no-proxy https://paddle-qa.bj.bcebos.com/PaddleHub/data/hub_data.tar
            tar -xzf hub_data.tar && mv hub_data/* . && rm -rf hub_data.tar

            wget -q --no-proxy https://paddle-qa.bj.bcebos.com/PaddleHub/hub_all_py.tar
            tar -xzf hub_all_py.tar && rm -rf hub_all_py.tar

            for hub_module in `cat hub_all_module.txt`
            do

            # 修改为CPU预测
            if [[ ${use_gpu} = False ]]; then
            sed -i "s/paddle.set_device('gpu')/paddle.set_device('cpu')/g" hub_all_py/test_${hub_module}.py
            sed -i "s/use_gpu=True/use_gpu=False/g" hub_all_py/test_${hub_module}.py
            fi

            echo ++++++++++++++++++++++ ${hub_module} start predicting !!!++++++++++++++++++++++
            $py_cmd hub_all_py/test_${hub_module}.py >> all_module_log/${hub_module}.log

            if [ $? -ne 0 ];then
                echo ++++++++++++++++++++++ ${hub_module} predict Failed!!!++++++++++++++++++++++
                hub_excption=$(expr ${hub_excption} + 1)
                hub_fail_list="${hub_fail_list} ${hub_module}"
                cat all_module_log/${hub_module}.log
                else
                echo ++++++++++++++++++++++ ${hub_module} predict Success!!!++++++++++++++++++++++
            fi

            sleep 5
            rm -rf /root/.paddlehub/modules/*
            rm -rf /root/.paddlehub/tmp
            done

            echo "================================== final-results =================================="
#            if [[ -e "log/whole_fail.log" ]]; then
#            cat log/whole_fail.log
#            fi
            echo "hub_excption = ${hub_excption}"
            echo "hub_fail_list is: ${hub_fail_list}"
            exit ${hub_excption}

        (*)
            echo "Error command"
            usage
            ;;
    esac
}

main

