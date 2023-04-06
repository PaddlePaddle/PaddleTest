set +x;
pwd;

####ce框架根目录
rm -rf ce && mkdir ce;
cd ce;

######################## 定义变量 ########################
# AGILE_PIPELINE_NAME 格式类似: PaddleClas-Linux-Cuda102-Python37-P0-Develop      (默认使用 Ubuntu 和用户保持一致)
# AGILE_PIPELINE_NAME 格式类似: PaddleClas-Linux-Cuda102-Python37-P0-Develop-Centos  (额外 Centos 和编包保持一致)
#其它内容或者可能不一致的不要随意加 "-", 下面是按照 "-" split 按序号填入的

#repo的名称
export reponame=${reponame:-"`(echo ${AGILE_PIPELINE_NAME}|awk -F '-' '{print $1}')`"}

#模型列表文件 , 固定路径及格式为 tools/reponame_优先级_list   优先级P2有多个用P21、P22  中间不用"-"划分, 防止按 "-" split 混淆
export models_file=${models_file:-"tools/${reponame}_`(echo ${AGILE_PIPELINE_NAME}|awk -F '-' '{print $5}')`_list"}
export models_list=${models_list:-None} #模型列表

#指定case操作系统
if [[ ${AGILE_PIPELINE_NAME} =~ "-Linux-" ]];then
    export system=${system:-"linux"}   # linux windows windows_cpu mac 与yaml case下字段保持一致
elif [[ ${AGILE_PIPELINE_NAME} =~ "-LinuxConvergence-" ]];then
    export system=${system:-"linux_convergence"}
elif [[ ${AGILE_PIPELINE_NAME} =~ "-Windows-" ]];then
    export system=${system:-"windows"}
elif [[ ${AGILE_PIPELINE_NAME} =~ "-WindowsCPU-" ]];then
    export system=${system:-"windows_cpu"}
elif [[ ${AGILE_PIPELINE_NAME} =~ "-Mac-" ]];then
    export system=${system:-"mac"}
else
    if [[ ${system} ]];then
        system=${system}
        ## 防止环境不匹配，不设置默认的 system
        # export system=${system:-"linux"}
    else
        echo "do not set system   or   AGILE_PIPELINE_NAME set inappropriate"
    fi
fi

#指定python版本
export Python_version=${Python_version:-"`(echo ${AGILE_PIPELINE_NAME}|awk -F 'Python' '{print $2}'|awk -F '-' '{print $1}')`"}

#指定docker镜像
if [[ ${AGILE_PIPELINE_NAME} =~ "Cuda102" ]];then
    if [[ ${AGILE_PIPELINE_NAME} =~ "Centos" ]];then
        export Image_version=${Image_version:-"registry.baidubce.com/paddlepaddle/paddle_manylinux_devel:cuda10.2-cudnn7.6-trt7.0-gcc8.2"}
    else
        export Image_version=${Image_version:-"registry.baidubce.com/paddlepaddle/paddle:latest-gpu-cuda10.2-cudnn7-dev"}
    fi
elif [[ ${AGILE_PIPELINE_NAME} =~ "Cuda112" ]];then
    if [[ ${AGILE_PIPELINE_NAME} =~ "Centos" ]];then
        export Image_version=${Image_version:-"registry.baidubce.com/paddlepaddle/paddle_manylinux_devel:cuda11.2-cudnn8.1-trt8.0-gcc8.2"}
    else
        export Image_version=${Image_version:-"registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda11.2-cudnn8.2-gcc82"}
    fi
elif [[ ${AGILE_PIPELINE_NAME} =~ "Cuda116" ]];then
    if [[ ${AGILE_PIPELINE_NAME} =~ "Centos" ]];then
        export Image_version=${Image_version:-"registry.baidubce.com/paddlepaddle/paddle_manylinux_devel:cuda11.6-cudnn8.4.0-trt8.4.0.6-gcc82"}
    else
        export Image_version=${Image_version:-"registry.baidubce.com/paddlepaddle/paddle:latest-dev-cuda11.6.2-cudnn8.4.0-gcc82"}
    fi
elif [[ ${AGILE_PIPELINE_NAME} =~ "Cuda117" ]];then
    if [[ ${AGILE_PIPELINE_NAME} =~ "Centos" ]];then
        export Image_version=${Image_version:-"registry.baidubce.com/paddlepaddle/paddle_manylinux_devel:cuda11.7-cudnn8.4-trt8.4-gcc8.2"}
    else
        export Image_version=${Image_version:-"registry.baidubce.com/paddlepaddle/paddleqa:latest-dev-cuda11.7-cudnn8.4-trt8.4-gcc8.2-v1"}
    fi
else
    if [[ ${Image_version} ]];then
        Image_version=${Image_version}
        ## 防止环境不匹配，不设置默认的 Image_version
        # export Image_version=${Image_version:-"registry.baidubce.com/paddlepaddle/paddle:latest-gpu-cuda10.2-cudnn7-dev"}
    else
        echo "do not set Image_version   or   AGILE_PIPELINE_NAME set inappropriate"
    fi
fi

#约定覆盖的几条流水线
#指定whl包, 暂时用night develop的包
if [[ ${AGILE_PIPELINE_NAME} =~ "Cuda102" ]] && [[ ${AGILE_PIPELINE_NAME} =~ "Python36" ]];then
    if [[ ${AGILE_PIPELINE_NAME} =~ "Develop" ]];then
        export paddle_whl=${paddle_whl:-"https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda10.2-cudnn7-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post102-cp36-cp36m-linux_x86_64.whl"}
    else
        export paddle_whl=${paddle_whl:-"https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda10.2-cudnn7-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post102-cp36-cp36m-linux_x86_64.whl"}
    fi
elif [[ ${AGILE_PIPELINE_NAME} =~ "Cuda102" ]] && [[ ${AGILE_PIPELINE_NAME} =~ "Python37" ]];then
    if [[ ${AGILE_PIPELINE_NAME} =~ "Develop" ]];then
        export paddle_whl=${paddle_whl:-"https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-GpuAll-LinuxCentos-Gcc82-Cuda102-Trtoff-Py37-Compile/latest/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl"}
    else
        export paddle_whl=${paddle_whl:-"https://paddle-qa.bj.bcebos.com/paddle-pipeline/Release-GpuAll-LinuxCentos-Gcc82-Cuda102-Trtoff-Py37-Compile/latest/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl"}
    fi
elif [[ ${AGILE_PIPELINE_NAME} =~ "Cuda112" ]] && [[ ${AGILE_PIPELINE_NAME} =~ "Python38" ]];then
    if [[ ${AGILE_PIPELINE_NAME} =~ "Develop" ]];then
        export paddle_whl=${paddle_whl:-"https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-GpuAll-LinuxCentos-Gcc82-Cuda112-Trtoff-Py38-Compile/latest/paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl"}
    else
        export paddle_whl=${paddle_whl:-"https://paddle-qa.bj.bcebos.com/paddle-pipeline/Release-GpuAll-LinuxCentos-Gcc82-Cuda112-Trton-Py38-Compile/latest/paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl"}
    fi
elif [[ ${AGILE_PIPELINE_NAME} =~ "Cuda116" ]] && [[ ${AGILE_PIPELINE_NAME} =~ "Python39" ]];then
    if [[ ${AGILE_PIPELINE_NAME} =~ "Develop" ]];then
        export paddle_whl=${paddle_whl:-"https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-TagBuild-Training-Linux-Gpu-Cuda11.6-Cudnn8-Mkl-Avx-Gcc8.2/latest/paddlepaddle_gpu-0.0.0.post116-cp39-cp39-linux_x86_64.whl"}
    else
        export paddle_whl=${paddle_whl:-"https://paddle-qa.bj.bcebos.com/paddle-pipeline/Release-TagBuild-Training-Linux-Gpu-Cuda11.6-Cudnn8-Mkl-Avx-Gcc8.2/latest/paddlepaddle_gpu-0.0.0.post116-cp39-cp39-linux_x86_64.whl"}
    fi
elif [[ ${AGILE_PIPELINE_NAME} =~ "Cuda117" ]] && [[ ${AGILE_PIPELINE_NAME} =~ "Python310" ]];then
    if [[ ${AGILE_PIPELINE_NAME} =~ "Develop" ]];then
        export paddle_whl=${paddle_whl:-"https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-TagBuild-Training-Linux-Gpu-Cuda11.7-Cudnn8-Mkl-Avx-Gcc8.2/latest/paddlepaddle_gpu-0.0.0.post117-cp310-cp310-linux_x86_64.whl"}
    else
        export paddle_whl=${paddle_whl:-"https://paddle-qa.bj.bcebos.com/paddle-pipeline/Release-TagBuild-Training-Linux-Gpu-Cuda11.7-Cudnn8-Mkl-Avx-Gcc8.2/latest/paddlepaddle_gpu-0.0.0.post117-cp310-cp310-linux_x86_64.whl"}
    fi
else
    if [[ ${paddle_whl} ]];then
        paddle_whl=${paddle_whl}
        ## 防止环境不匹配，不设置默认的paddle_whl
        # if [[ ${AGILE_PIPELINE_NAME} =~ "Develop" ]];then
        #     export paddle_whl=${paddle_whl:-"https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda10.2-cudnn7-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post102-cp37-cp37m-linux_x86_64.whl"}
        # else
        #     export paddle_whl=${paddle_whl:-"https://paddle-wheel.bj.bcebos.com/develop/linux/linux-gpu-cuda10.2-cudnn7-mkl-gcc8.2-avx/paddlepaddle_gpu-0.0.0.post102-cp37-cp37m-linux_x86_64.whl"}
        # fi
    else
        echo "do not set paddle_whl   or   AGILE_PIPELINE_NAME set inappropriate"
    fi
fi


#### 可能要改的参数
export step=${step:-train}  #阶段 demo:train:multi,single+eval:trained,pretrained, 所有流水线都要自己改
export branch=${branch:-develop}    # repo的分支，大部分为develop，如果有master dygraph等注意设置!!
export mode=${mode:-function}   #function只验证功能是否正常  precision验证功能&小数据集精度
export timeout=${timeout:-72000}   #timeout 为超时取消时间, 单位为秒
export docker_flag=${docker_flag:-} # 如果北京集群cce环境为False，自己的开发机&release机器不用设置

#### 建议不改的参数
export use_build=${use_build:-yes}  #流水线默认为yes，是否在main中执行环境部署
export get_repo=${get_repo:-wget} #现支持10个库，需要的话可以加，wget快很多
export set_cuda=${set_cuda:-} #预先不设置   #设置显卡号，流水线不用设置，后面有通过release_01后缀判断

#### 数据软链使用
export dataset_org=${dataset_org:-None}     #如需软链数据基于根目录的原始地址 demo: /ssd2/ce_data
export dataset_target=${dataset_target:-None}   #如需软链数据基于reponame的目标地址 demo: data/flower102

#### 全局设置的参数
export no_proxy=${no_proxy:-}
export http_proxy=${http_proxy:-}   # 代理在效率云全局变量设置
export AGILE_PIPELINE_CONF_ID=${AGILE_PIPELINE_CONF_ID}   #效率云依赖参数
export AGILE_PIPELINE_BUILD_ID=${AGILE_PIPELINE_BUILD_ID} #效率云依赖参数
export AGILE_JOB_BUILD_ID=${AGILE_JOB_BUILD_ID}   #效率云依赖参数
export AGILE_WORKSPACE=${AGILE_WORKSPACE}   #效率云依赖参数
export REPORT_SERVER_PASSWORD=${REPORT_SERVER_PASSWORD}   #上传全局变量

#### 根据PaddleTest & 框架名称决定的参数
export CE_version_name=${CE_version_name:-TestFrameWork}    #与测试框架的名称一致
export models_name=${models_name:-models_restruct}  #后面复制使用，和模型库的父路径目录一致（后续改为models）



######################## 开始执行 ########################
####    测试框架下载    #####
wget -q ${CE_Link}  --no-check-certificate #需要全局定义
unzip -P ${CE_pass} ${CE_version_name}.zip

####设置代理  proxy不单独配置 表示默认有全部配置，不用export
export http_proxy=${http_proxy}
export https_proxy=${http_proxy}
export no_proxy=${no_proxy}
export AK=${AK} #使用bos_new上传需要
export SK=${SK}
export bce_whl_url=${bce_whl_url}
set -x;

#输出参数验证
echo "@@@reponame: ${reponame}"
echo "@@@models_list: ${models_list}"
echo "@@@models_file: ${models_file}"
echo "@@@system: ${system}"
echo "@@@Python_version: ${Python_version}"
echo "@@@Image_version: ${Image_version}"
echo "@@@paddle_whl: ${paddle_whl}"
echo "@@@step: ${step}"
echo "@@@branch: ${branch}"
echo "@@@mode: ${mode}"
echo "@@@docker_flag: ${docker_flag}"
echo "@@@timeout: ${timeout}"

####之前下载过了直接mv
if [[ -d "../task" ]];then
    mv ../task .  #如果预先下载直接mv
else
    wget -q https://xly-devops.bj.bcebos.com/PaddleTest/PaddleTest.tar.gz --no-proxy  --no-check-certificate >/dev/null
    tar xf PaddleTest.tar.gz >/dev/null 2>&1
    mv PaddleTest task
fi

#复制模型相关文件到指定位置
cp -r ./task/${models_name}/${reponame}/.  ./${CE_version_name}/
ls ./${CE_version_name}/
cd ./${CE_version_name}/

##如果预先模型库下载直接mv, 方便二分是checkout 到某个commit进行二分
if [[ -d "../../${reponame}" ]];then  #前面cd 了 2次所以使用 ../../
    cp -r ../../${reponame} .
    echo "因为 ${reponame} 在根目录存在 使用预先clone或wget的 ${reponame}"
fi

####根据agent制定对应卡，记得起agent时文件夹按照release_01 02 03 04名称
if  [[ "${set_cuda}" == "" ]] ;then  #换了docker启动的方式，使用默认制定方式即可，SET_MULTI_CUDA参数只是在启动时使用
    tc_name=`(echo $PWD|awk -F 'xly/' '{print $2}'|awk -F '/' '{print $1}')`
    echo "teamcity path:" $tc_name
    if [ $tc_name == "release_02" ];then
        echo release_02
        export set_cuda=1;
        if [[ "${docker_flag}" == "" ]]; then
            fuser -v /dev/nvidia1 | awk '{print $0}' | xargs kill -9
        fi
    elif [ $tc_name == "release_03" ];then
        echo release_02
        export set_cuda=2;
        if [[ "${docker_flag}" == "" ]]; then
            fuser -v /dev/nvidia2 | awk '{print $0}' | xargs kill -9
        fi
    elif [ $tc_name == "release_04" ];then
        echo release_04
        export set_cuda=3;
        if [[ "${docker_flag}" == "" ]]; then
            fuser -v /dev/nvidia3 | awk '{print $0}' | xargs kill -9
        fi
    elif [ $tc_name == "release_05" ];then
        echo release_05
        export set_cuda=4;
        if [[ "${docker_flag}" == "" ]]; then
            fuser -v /dev/nvidia4 | awk '{print $0}' | xargs kill -9
        fi
    elif [ $tc_name == "release_06" ];then
        echo release_06
        export set_cuda=5;
        if [[ "${docker_flag}" == "" ]]; then
            fuser -v /dev/nvidia5 | awk '{print $0}' | xargs kill -9
        fi
    elif [ $tc_name == "release_07" ];then
        echo release_07
        export set_cuda=6;
        if [[ "${docker_flag}" == "" ]]; then
            fuser -v /dev/nvidia6 | awk '{print $0}' | xargs kill -9
        fi
    elif [ $tc_name == "release_08" ];then
        echo release_08
        export set_cuda=7;
        if [[ "${docker_flag}" == "" ]]; then
            fuser -v /dev/nvidia7 | awk '{print $0}' | xargs kill -9
        fi
    else
        echo release_01
        export set_cuda=0;
        if [[ "${docker_flag}" == "" ]]; then
            fuser -v /dev/nvidia0 | awk '{print $0}' | xargs kill -9
        fi
    fi
else
    echo already seted CUDA_id  #这里需要再细化下，按下面的方法指定无用，直接默认按common中指定0,1卡了
    export set_cuda=${set_cuda}
fi

if [[ "${docker_flag}" == "" ]]; then

    echo "before set_cuda: $set_cuda" #在docker外更改set_cuda从0开始计数, 在cce或线下已在docker中不执行
    export set_cuda_back=${set_cuda};
    array=(${set_cuda_back//,/ });
    set_cuda=0;
    for((i=1;i<${#array[@]};i++));
    do
    export set_cuda=${set_cuda},${i};
    done
    echo "after set_cuda: $set_cuda"

    ####创建docker
    set +x;
    docker_name="ce_${AGILE_PIPELINE_NAME}_${AGILE_JOB_BUILD_ID}" #AGILE_JOB_BUILD_ID以每个流水线粒度区分docker名称
    function docker_del()
    {
    echo "begin kill docker"
    docker rm -f ${docker_name}
    echo "end kill docker"
    }
    trap 'docker_del' SIGTERM
    ## 使用修改之前的set_cuda_back
    NV_GPU=${set_cuda_back} nvidia-docker run -i   --rm \
        --name=${docker_name} --net=host \
        --shm-size=128G \
        -v $(pwd):/workspace \
        -v /home/:/home/ \
        -v /mnt:/mnt  \
        -v /ssd3/dy2st_data/:/ssd3/dy2st_data/  \
        -e AK=${AK} \
        -e SK=${SK} \
        -e bce_whl_url=${bce_whl_url} \
        -e PORT_RANGE="62000:65536" \
        -e no_proxy=${no_proxy} \
        -e http_proxy=${http_proxy} \
        -e https_proxy=${https_proxy} \
        -e AGILE_PIPELINE_CONF_ID=${AGILE_PIPELINE_CONF_ID} \
        -e AGILE_PIPELINE_BUILD_ID=${AGILE_PIPELINE_BUILD_ID} \
        -e AGILE_JOB_BUILD_ID=${AGILE_JOB_BUILD_ID} \
        -e AGILE_PIPELINE_NAME=${AGILE_PIPELINE_NAME} \
        -e AGILE_WORKSPACE=${AGILE_WORKSPACE} \
        -e REPORT_SERVER_PASSWORD=${REPORT_SERVER_PASSWORD} \
        -e Python_version=${Python_version} \
        -e models_list=${models_list} \
        -e models_file=${models_file} \
        -e system=${system} \
        -e step=${step} \
        -e reponame=${reponame} \
        -e timeout=${timeout} \
        -e mode=${mode} \
        -e use_build=${use_build} \
        -e branch=${branch} \
        -e get_repo=${get_repo} \
        -e paddle_whl=${paddle_whl} \
        -e dataset_org=${dataset_org} \
        -e dataset_target=${dataset_target} \
        -e set_cuda=${set_cuda} \
        -w /workspace \
        ${Image_version}  \
        /bin/bash -c '

        ldconfig;
        if [[ `yum --help` =~ "yum" ]];then
            echo "centos"
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
        else
            echo "ubuntu"
            case ${Python_version} in
            36)
            mkdir run_env_py36;
            ln -s $(which python3.6) run_env_py36/python;
            ln -s $(which pip3.6) run_env_py36/pip;
            export PATH=$(pwd)/run_env_py36:${PATH};
            ;;
            37)
            mkdir run_env_py37;
            ln -s $(which python3.7) run_env_py37/python;
            ln -s $(which pip3.7) run_env_py37/pip;
            export PATH=$(pwd)/run_env_py37:${PATH};
            ;;
            38)
            mkdir run_env_py38;
            ln -s $(which python3.8) run_env_py38/python;
            ln -s $(which pip3.8) run_env_py38/pip;
            export PATH=$(pwd)/run_env_py38:${PATH};
            ;;
            39)
            mkdir run_env_py39;
            ln -s $(which python3.9) run_env_py39/python;
            ln -s $(which pip3.9) run_env_py39/pip;
            export PATH=$(pwd)/run_env_py39:${PATH};
            ;;
            310)
            mkdir run_env_py310;
            ln -s $(which python3.10) run_env_py310/python;
            ln -s $(which pip3.10) run_env_py310/pip;
            export PATH=$(pwd)/run_env_py310:${PATH};
            ;;
            esac
        fi

        nvidia-smi;
        python -c "import sys; print(sys.version_info[:])";
        git --version;
        #解决编译BUG,加mkdir
        mkdir -p /paddle/build/third_party/CINN/src/external_cinn-build/thirds/install/jitify/stl_headers/algorithm
        python -m pip install --user -U pip  -i https://mirror.baidu.com/pypi/simple #升级pip
        python -m pip install --user -U -r requirements.txt  -i https://mirror.baidu.com/pypi/simple #预先安装依赖包
        python main.py --models_list=${models_list:-None} --models_file=${models_file:-None} --system=${system:-linux} --step=${step:-train} --reponame=${reponame:-PaddleClas} --mode=${mode:-function} --use_build=${use_build:-yes} --branch=${branch:-develop} --get_repo=${get_repo:-wget} --paddle_whl=${paddle_whl:-None} --dataset_org=${dataset_org:-None} --dataset_target=${dataset_target:-None} --set_cuda=${set_cuda:-0,1} --timeout=${timeout:-3600}
    ' &
    wait $!
    exit $?
fi
