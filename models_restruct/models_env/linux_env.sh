set +x;
pwd;

####ce框架根目录
rm -rf ce && mkdir ce;
cd ce;

#### 预设默认参数
export models_list=${models_list:-None} #模型列表
export models_file=${models_file:-None} #模型列表文件   #预先不设置，二选一
export system=${system:-linux}  # linux windows windows_cpu mac 与yaml case下字段保持一致
export step=${step:-train}  #阶段 demo:train:multi,single+eval:trained,pretrained
export reponame=${reponame:-PaddleClas} #repo的名称
export branch=${branch:-develop}    # repo的分支，大部分为develop，如果有master dygraph等注意设置!!
export mode=${mode:-function}   #function只验证功能是否正常  precision验证功能&小数据集精度
export use_build=${use_build:-yes}  #流水线默认为yes，是否在main中执行环境部署
export get_repo=${get_repo:-wget} #现支持10个库，需要的话可以加，wget快很多
export paddle_whl=${paddle_whl:-None}   #paddlewhl包地址，为None则认为已安装不用安装
export dataset_org=${dataset_org:-None}     #如需软链数据基于根目录的原始地址 demo: /ssd2/ce_data
export dataset_target=${dataset_target:-None}   #如需软链数据基于reponame的目标地址 demo: data/flower102
export set_cuda=${set_cuda:-} #预先不设置   #设置显卡号，流水线不用设置，后面有通过release_01后缀判断

#额外的变量
export AGILE_PIPELINE_CONF_ID=$AGILE_PIPELINE_CONF_ID   #效率云依赖参数
export AGILE_PIPELINE_BUILD_ID=$AGILE_PIPELINE_BUILD_ID #效率云依赖参数
export AGILE_JOB_BUILD_ID=$AGILE_JOB_BUILD_ID   #效率云依赖参数
export docker_flag=${docker_flag:-} # 如果北京集群cce环境为False，自己的开发机不用设置
export http_proxy=${http_proxy:-}   # 代理在效率云全局变量设置
export no_proxy=${no_proxy:-}
export Python_env=${Python_env:-path_way}   # manylinux使用 path_way  paddle:latest使用 ln_way
export Python_version=${Python_version:-37} # 指定ython版本
export Image_version=${Image_version:-registry.baidubce.com/paddlepaddle/paddle_manylinux_devel:cuda10.2-cudnn7}
#指定docker版本

# 预设一些可能会修改的变量
export CE_version_name=${CE_version_name:-TestFrameWork}    #与测试框架的名称一致
export models_name=${models_name:-models_restruct}  #后面复制使用，和模型库的父路径目录一致（后续改为models）

####    测试框架下载    #####
wget -q ${CE_Link} #需要全局定义
unzip -P ${CE_pass} ${CE_version_name}.zip

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
####之前下载过了直接mv
if [[ -d "../task" ]];then
    mv ../task .  #如果预先下载直接mv
else
    wget -q https://xly-devops.bj.bcebos.com/PaddleTest/PaddleTest.tar.gz --no-proxy  >/dev/null
    tar xf PaddleTest.tar.gz >/dev/null 2>&1
    mv PaddleTest task
fi

#复制模型相关文件到指定位置
cp -r ./task/${models_name}/${reponame}/.  ./${CE_version_name}/
ls ./${CE_version_name}/
cd ./${CE_version_name}/

####根据agent制定对应卡，记得起agent时文件夹按照release_01 02 03 04名称  ##TODO:暂时先考虑两张卡，后续优化
if  [[ "${set_cuda}" == "" ]] ;then  #换了docker启动的方式，使用默认制定方式即可，SET_MULTI_CUDA参数只是在启动时使用
    tc_name=`(echo $PWD|awk -F '/' '{print $4}')`
    echo "teamcity path:" $tc_name
    if [ $tc_name == "release_02" ];then
        echo release_02
        export set_cuda=2,3;

    elif [ $tc_name == "release_03" ];then
        echo release_03
        export set_cuda=4,5;

    elif [ $tc_name == "release_04" ];then
        echo release_04
        export set_cuda=6,7;
    else
        echo release_01
        export set_cuda=0,1;
    fi
else
    echo already seted CUDA_id  #这里需要再细化下，按下面的方法指定无用，直接默认按common中指定0,1卡了
    export set_cuda=${set_cuda}
fi

if [[ "${docker_flag}" == "" ]]; then
    ####创建docker
    set +x;
    docker_name="ce_${reponame}_${AGILE_JOB_BUILD_ID}" #AGILE_JOB_BUILD_ID以每个流水线粒度区分docker名称
    function docker_del()
    {
    echo "begin kill docker"
    docker rm -f ${docker_name}
    echo "end kill docker"
    }
    trap 'docker_del' SIGTERM
    NV_GPU=${set_cuda} nvidia-docker run -i   --rm \
        --name=${docker_name} --net=host \
        --shm-size=128G \
        -v $(pwd):/workspace \
        -v /ssd2:/ssd2 \
        -w /workspace \
        ${Image_version}  \
        /bin/bash -c "

        ldconfig;
        #额外的变量
        export no_proxy=${no_proxy};
        export http_proxy=${http_proxy};
        export https_proxy=${http_proxy};
        export AGILE_PIPELINE_CONF_ID=$AGILE_PIPELINE_CONF_ID;
        export AGILE_PIPELINE_BUILD_ID=$AGILE_PIPELINE_BUILD_ID;
        export AGILE_JOB_BUILD_ID=$AGILE_JOB_BUILD_ID;

        if [[ ${Python_env} == 'ln_way' ]];then
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
        fi

        nvidia-smi;
        python -c 'import sys; print(sys.version_info[:])';
        git --version;
        python -m pip install -r requirements.txt #预先安装依赖包
        python main.py --models_list=${models_list:-None} --models_file=${models_file:-None} --system=${system:-linux} --step=${step:-train} --reponame=${reponame:-PaddleClas} --mode=${mode:-function} --use_build=${use_build:-yes} --branch=${branch:-develop} --get_repo=${get_repo:-wget} --paddle_whl=${paddle_whl:-None} --dataset_org=${dataset_org:-None} --dataset_target=${dataset_target:-None}
    " &
    wait $!
    exit $?
else
    ldconfig;
    if [[ ${Python_env} == 'ln_way' ]];then
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
    fi

    nvidia-smi;
    python -c 'import sys; print(sys.version_info[:])';
    git --version;
    python -m pip install -r requirements.txt #预先安装依赖包
    python main.py --models_list=${models_list:-None} --models_file=${models_file:-None} --system=${system:-linux} --step=${step:-train} --reponame=${reponame:-PaddleClas} --mode=${mode:-function} --use_build=${use_build:-yes} --branch=${branch:-develop} --get_repo=${get_repo:-wget} --paddle_whl=${paddle_whl:-None} --dataset_org=${dataset_org:-None} --dataset_target=${dataset_target:-None}
fi
