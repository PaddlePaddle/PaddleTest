set +x;
pwd;

####ce框架根目录
rm -rf ce && mkdir ce;
cd ce;

########TODO：区分下是否使用CE框架、区分下是否单独clone库(用于CI)、步骤Common_name需要再细化一下

export Project_path=${Project_path:-/workspace/task/PaddleClas}
export Data_path=${Data_path:-/ssd2/ce_data/PaddleClas}
export Repo=${Repo:-PaddleClas}
export Python_env=${Python_env:-path_way}
export Python_version=${Python_version:-37}
export CE_version=${CE_version:-V1}
export Priority_version=${Priority_version:-P0}
export Compile_version=${Compile_version:-https://paddle-qa.bj.bcebos.com/paddle-pipeline/Release-GpuAll-LinuxCentos-Gcc82-Cuda102-Trtoff-Py37-Compile/latest/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl}
export Image_version=${Image_version:-registry.baidubce.com/paddlepaddle/paddle_manylinux_devel:cuda10.2-cudnn7}
export Common_name=${Common_name:-conf/cls_common}  #CE框架中的执行步骤，名称各异所以需要传入

export SET_MULTI_CUDA=${SET_MULTI_CUDA:-}  #如果不使用流水线，手动设置卡号 默认不设置，用0 1卡
export docker_flag=${docker_flag:-}  #是否在docker内的环境 默认不设置，如果在docker中进行设置为False
#手动设置卡号

####测试框架下载
if [[ ${CE_version} == "V2" ]];then
    export CE_version_name=continuous_evaluation
    wget -q ${CE_V2}
else
    export CE_version_name=Paddle_Cloud_CE
    wget -q ${CE_V1}
fi
ls
unzip -P ${CE_pass}  ${CE_version_name}.zip

####设置代理  proxy不单独配置 表示默认有全部配置，不用export
if  [[ ! -n "${http_proxy}" ]] ;then
    echo unset http_proxy
    export http_proxy=${http_proxy}
    export https_proxy=${http_proxy}
else
    export http_proxy=${http_proxy}
    export https_proxy=${http_proxy}
fi
export no_proxy=${no_proxy}
set -x;
ls;

####之前下载过了直接mv
if [[ -d "../task" ]];then
    mv ../task .  #task路径是CE框架写死的
else
    wget -q https://xly-devops.bj.bcebos.com/PaddleTest/PaddleTest.tar.gz --no-proxy  >/dev/null
    tar xf PaddleTest.tar.gz >/dev/null 2>&1
    mv PaddleTest task
fi

#通用变量[用户改]
test_code_download_path=./task/models/${Repo}

#迁移下载路径代码和配置到框架指定执行路径 [不用改]
mkdir -p ${test_code_download_path}/log
ls ${test_code_download_path}/log;
cp -r ${test_code_download_path}/.  ./${CE_version_name}/src/task
cp ${test_code_download_path}/${Common_name}.py ./${CE_version_name}/src/task/common.py
cat ./${CE_version_name}/src/task/common.py;
ls;

####根据agent制定对应卡，记得起agent时文件夹按照release_01 02 03 04名称
if  [[ "${SET_MULTI_CUDA}" == "" ]] ;then  #换了docker启动的方式，使用默认制定方式即可，SET_MULTI_CUDA参数只是在启动时使用
    tc_name=`(echo $PWD|awk -F '/' '{print $4}')`
    echo "teamcity path:" $tc_name
    if [ $tc_name == "release_02" ];then
        echo release_02
        export SET_MULTI_CUDA=2,3;

    elif [ $tc_name == "release_03" ];then
        echo release_03
        export SET_MULTI_CUDA=4,5;

    elif [ $tc_name == "release_04" ];then
        echo release_04
        export SET_MULTI_CUDA=6,7;
    else
        echo release_01
        export SET_MULTI_CUDA=0,1;
    fi
else
    echo already seted CUDA_id  #这里需要再细化下，按下面的方法指定无用，直接默认按common中指定0,1卡了
    export SET_CUDA=${SET_MULTI_CUDA}
    export SET_MULTI_CUDA=${SET_MULTI_CUDA}
fi
####显示执行步骤
cat ./${CE_version_name}/src/task/common.py

#####进入执行路径创建docker容器 [用户改docker创建]
cd ./${CE_version_name}/src/task
ls;
wget -q https://xly-devops.bj.bcebos.com/PaddleTest/PaddleClas.tar.gz --no-proxy  >/dev/null
#预先下载PaddleClas，不使用CE框架clone
tar xf PaddleClas.tar.gz
rm -rf PaddleClas.tar.gz
cd ..
ls;

#cd ${CE_version_name}/src
# ls

if [[ "${docker_flag}" == "" ]]; then

    #升级显卡策略，独立使用显卡，以逗号分割执行显卡编号，重定义从0开始赋值  这种情况是适配大于两张卡的情况
    # echo SET_CUDA VS SET_MULTI_CUDA
    # echo $SET_CUDA
    # echo $SET_MULTI_CUDA
    # export SET_CUDA=0;
    # SET_MULTI_CUDA_back=${SET_MULTI_CUDA};
    # array=(${SET_MULTI_CUDA_back//,/ });
    # SET_MULTI_CUDA=0;
    # for((i=1;i<${#array[@]};i++));
    # do
    # export SET_MULTI_CUDA=${SET_MULTI_CUDA},${i};
    # done
    # echo $SET_CUDA
    # echo $SET_MULTI_CUDA

    ####创建docker
    set +x;
    docker_name="ce_${Repo}_${Priority_version}_${AGILE_JOB_BUILD_ID}" #AGILE_JOB_BUILD_ID以每个流水线粒度区分docker名称
    function docker_del()
    {
    echo "begin kill docker"
    docker rm -f ${docker_name}
    echo "end kill docker"
    }
    trap 'docker_del' SIGTERM
    # NV_GPU=${SET_MULTI_CUDA_back} nvidia-docker run -i   --rm \
    NV_GPU=${SET_MULTI_CUDA} nvidia-docker run -i   --rm \
                --name=${docker_name} --net=host \
                --shm-size=128G \
                -v $(pwd):/workspace \
                -v /ssd2:/ssd2 \
                -w /workspace \
                ${Image_version}  \
                /bin/bash -c "

                ldconfig;
                export no_proxy=${no_proxy};
                export http_proxy=${http_proxy};
                export https_proxy=${http_proxy};
                export Data_path=${Data_path};
                export Project_path=${Project_path};
                # export SET_CUDA=${SET_CUDA};
                # export SET_MULTI_CUDA=${SET_MULTI_CUDA};

                if [[ ${Python_env} == 'path_way' ]];then
                    case ${Python_version} in
                    36)
                    export LD_LIBRARY_PATH=/opt/_internal/cpython-3.6.0/lib/:${LD_LIBRARY_PATH}
                    export PATH=/opt/_internal/cpython-3.6.0/bin/:${PATH}
                    ;;
                    37)
                    export LD_LIBRARY_PATH=/opt/_internal/cpython-3.7.0/lib/:${LD_LIBRARY_PATH}
                    export PATH=/opt/_internal/cpython-3.7.0/bin/:${PATH}
                    ;;
                    38)
                    export LD_LIBRARY_PATH=/opt/_internal/cpython-3.8.0/lib/:${LD_LIBRARY_PATH}
                    export PATH=/opt/_internal/cpython-3.8.0/bin/:${PATH}
                    ;;
                    39)
                    export LD_LIBRARY_PATH=/opt/_internal/cpython-3.9.0/lib/:${LD_LIBRARY_PATH}
                    export PATH=/opt/_internal/cpython-3.9.0/bin/:${PATH}
                    ;;
                    310)
                    export LD_LIBRARY_PATH=/opt/_internal/cpython-3.10.0/lib/:${LD_LIBRARY_PATH}
                    export PATH=/opt/_internal/cpython-3.10.0/bin/:${PATH}
                    ;;
                    esac
                elif [[ ${Python_env} == 'ln_way' ]];then
                    # rm -rf /usr/bin/python2.7
                    # rm -rf /usr/local/python2.7.15/bin/python
                    # rm -rf /usr/local/bin/python
                    # export PATH=/usr/local/bin/python:${PATH}
                    case ${Python_version} in
                    36)
                    # ln -s /usr/local/bin/python3.6 /usr/local/bin/python
                    mkdir run_env_py36;
                    ln -s $(which python3.6) run_env_py36/python;
                    ln -s $(which pip3.6) run_env_py36/pip;
                    export PATH=$(pwd)/run_env_py36:${PATH};
                    ;;
                    37)
                    # ln -s /usr/local/bin/python3.7 /usr/local/bin/python
                    mkdir run_env_py37;
                    ln -s $(which python3.7) run_env_py37/python;
                    ln -s $(which pip3.7) run_env_py37/pip;
                    export PATH=$(pwd)/run_env_py37:${PATH};
                    ;;
                    38)
                    # ln -s /usr/local/bin/python3.8 /usr/local/bin/python
                    mkdir run_env_py38;
                    ln -s $(which python3.8) run_env_py38/python;
                    ln -s $(which pip3.8) run_env_py38/pip;
                    export PATH=$(pwd)/run_env_py38:${PATH};
                    ;;
                    39)
                    # ln -s /usr/local/bin/python3.9 /usr/local/bin/python
                    mkdir run_env_py39;
                    ln -s $(which python3.9) run_env_py39/python;
                    ln -s $(which pip3.9) run_env_py39/pip;
                    export PATH=$(pwd)/run_env_py39:${PATH};
                    ;;
                    310)
                    # ln -s /usr/local/bin/python3.10 /usr/local/bin/python
                    mkdir run_env_py310;
                    ln -s $(which python3.10) run_env_py310/python;
                    ln -s $(which pip3.10) run_env_py310/pip;
                    export PATH=$(pwd)/run_env_py310:${PATH};
                    ;;
                    esac
                else
                    echo unset python version;
                fi

                nvidia-smi;
                python -c 'import sys; print(sys.version_info[:])';
                git --version;
                if [[ ${CE_version} == 'V2' ]];then
                    bash main.sh --build_id=${AGILE_PIPELINE_BUILD_ID} --build_type_id=${AGILE_PIPELINE_CONF_ID} --priority=${Priority_version} --compile_path=${Compile_version} --job_build_id=${AGILE_JOB_BUILD_ID};
                else
                    bash main.sh --task_type='model' --build_number=${AGILE_PIPELINE_BUILD_NUMBER} --project_name=${AGILE_MODULE_NAME} --task_name=${AGILE_PIPELINE_NAME}  --build_id=${AGILE_PIPELINE_BUILD_ID} --build_type=${AGILE_PIPELINE_UUID} --owner='paddle' --priority=${Priority_version} --compile_path=${Compile_version} --agile_job_build_id=${AGILE_JOB_BUILD_ID};
                fi
    " &
    wait $!
    exit $?
else
    ldconfig;
    if [[ ${Python_env} == 'path_way' ]];then
        case ${Python_version} in
        36)
        export LD_LIBRARY_PATH=/opt/_internal/cpython-3.6.0/lib/:${LD_LIBRARY_PATH}
        export PATH=/opt/_internal/cpython-3.6.0/bin/:${PATH}
        ;;
        37)
        export LD_LIBRARY_PATH=/opt/_internal/cpython-3.7.0/lib/:${LD_LIBRARY_PATH}
        export PATH=/opt/_internal/cpython-3.7.0/bin/:${PATH}
        ;;
        38)
        export LD_LIBRARY_PATH=/opt/_internal/cpython-3.8.0/lib/:${LD_LIBRARY_PATH}
        export PATH=/opt/_internal/cpython-3.8.0/bin/:${PATH}
        ;;
        39)
        export LD_LIBRARY_PATH=/opt/_internal/cpython-3.9.0/lib/:${LD_LIBRARY_PATH}
        export PATH=/opt/_internal/cpython-3.9.0/bin/:${PATH}
        ;;
        310)
        export LD_LIBRARY_PATH=/opt/_internal/cpython-3.10.0/lib/:${LD_LIBRARY_PATH}
        export PATH=/opt/_internal/cpython-3.10.0/bin/:${PATH}
        ;;
        esac
    elif [[ ${Python_env} == 'ln_way' ]];then
        # rm -rf /usr/bin/python2.7
        # rm -rf /usr/local/python2.7.15/bin/python
        # rm -rf /usr/local/bin/python
        # export PATH=/usr/local/bin/python:${PATH}
        case ${Python_version} in
        36)
        # ln -s /usr/local/bin/python3.6 /usr/local/bin/python
        mkdir run_env_py36;
        ln -s $(which python3.6) run_env_py36/python;
        ln -s $(which pip3.6) run_env_py36/pip;
        export PATH=$(pwd)/run_env_py36:${PATH};
        ;;
        37)
        # ln -s /usr/local/bin/python3.7 /usr/local/bin/python
        mkdir run_env_py37;
        ln -s $(which python3.7) run_env_py37/python;
        ln -s $(which pip3.7) run_env_py37/pip;
        export PATH=$(pwd)/run_env_py37:${PATH};
        ;;
        38)
        # ln -s /usr/local/bin/python3.8 /usr/local/bin/python
        mkdir run_env_py38;
        ln -s $(which python3.8) run_env_py38/python;
        ln -s $(which pip3.8) run_env_py38/pip;
        export PATH=$(pwd)/run_env_py38:${PATH};
        ;;
        39)
        # ln -s /usr/local/bin/python3.9 /usr/local/bin/python
        mkdir run_env_py39;
        ln -s $(which python3.9) run_env_py39/python;
        ln -s $(which pip3.9) run_env_py39/pip;
        export PATH=$(pwd)/run_env_py39:${PATH};
        ;;
        310)
        # ln -s /usr/local/bin/python3.10 /usr/local/bin/python
        mkdir run_env_py310;
        ln -s $(which python3.10) run_env_py310/python;
        ln -s $(which pip3.10) run_env_py310/pip;
        export PATH=$(pwd)/run_env_py310:${PATH};
        ;;
        esac
    else
        echo unset python version;
    fi

    python -c 'import sys; print(sys.version_info[:])';
    git --version;
    if [[ ${CE_version} == 'V2' ]];then
        bash main.sh --build_id=${AGILE_PIPELINE_BUILD_ID} --build_type_id=${AGILE_PIPELINE_CONF_ID} --priority=${Priority_version} --compile_path=${Compile_version} --job_build_id=${AGILE_JOB_BUILD_ID};
    else
        bash main.sh --task_type='model' --build_number=${AGILE_PIPELINE_BUILD_NUMBER} --project_name=${AGILE_MODULE_NAME} --task_name=${AGILE_PIPELINE_NAME}  --build_id=${AGILE_PIPELINE_BUILD_ID} --build_type=${AGILE_PIPELINE_UUID} --owner='paddle' --priority=${Priority_version} --compile_path=${Compile_version} --agile_job_build_id=${AGILE_JOB_BUILD_ID};
    fi
fi
