set +x;
pwd;

####ce框架根目录
rm -rf ce && mkdir ce;
cd ce;

#### 预设默认参数
export models_list=${models_list:-None}
export models_file=${models_file:-None} #预先不设置，二选一
export system=${system:-linux}
export step=${step:-train}
export reponame=${reponame:-PaddleClas}
export mode=${mode:-function}
export use_build=${use_build:-yes}
export branch=${branch:-develop}
export get_repo=${get_repo:-wget} #现支持10个库，需要的话可以加，wget快很多
export paddle_whl=${paddle_whl:-None}
export dataset_org=${dataset_org:-None}
export dataset_target=${dataset_target:-None}
export set_cuda=${set_cuda:-} #预先不设置

#额外的变量
export docker_flag=${docker_flag:-}
export http_proxy=${http_proxy:-}
export no_proxy=${no_proxy:-}
export Python_env=${Python_env:-ln_way}
#后续docker都是用paddlepaddle(ln_way)，不用manylinux(path_way)
export Python_version=${Python_version:-37}
export Image_version=${Image_version:-37}

# 预设一些可能会修改的变量
export CE_version_name=${CE_version_name:-TestFrameWork}
export models_name=${models_name:-models_restruct}

####测试框架下载
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
ls CE_version_name

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
    # NV_GPU=${SET_MULTI_CUDA_back} nvidia-docker run -i   --rm \
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

        if [[ ${Python_env} == 'ln_way' ]];then
            rm -rf /usr/bin/python2.7
            rm -rf /usr/local/python2.7.15/bin/python
            rm -rf /usr/local/bin/python
            export PATH=/usr/local/bin/python:${PATH}
            case ${Python_version} in
            36)
            ln -s /usr/local/bin/python3.6 /usr/local/bin/python
            # mkdir run_env_py36;
            # ln -s $(which python3.6) run_env_py36/python;
            # ln -s $(which pip3.6) run_env_py36/pip;
            # export PATH=$(pwd)/run_env_py36:${PATH};
            ;;
            37)
            ln -s /usr/local/bin/python3.7 /usr/local/bin/python
            # mkdir run_env_py37;
            # ln -s $(which python3.7) run_env_py37/python;
            # ln -s $(which pip3.7) run_env_py37/pip;
            # export PATH=$(pwd)/run_env_py37:${PATH};
            ;;
            38)
            ln -s /usr/local/bin/python3.8 /usr/local/bin/python
            # mkdir run_env_py38;
            # ln -s $(which python3.8) run_env_py38/python;
            # ln -s $(which pip3.8) run_env_py38/pip;
            # export PATH=$(pwd)/run_env_py38:${PATH};
            ;;
            39)
            ln -s /usr/local/bin/python3.9 /usr/local/bin/python
            # mkdir run_env_py39;
            # ln -s $(which python3.9) run_env_py39/python;
            # ln -s $(which pip3.9) run_env_py39/pip;
            # export PATH=$(pwd)/run_env_py39:${PATH};
            ;;
            310)
            ln -s /usr/local/bin/python3.10 /usr/local/bin/python
            # mkdir run_env_py310;
            # ln -s $(which python3.10) run_env_py310/python;
            # ln -s $(which pip3.10) run_env_py310/pip;
            # export PATH=$(pwd)/run_env_py310:${PATH};
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
        python main.py models_list=${models_list:-None} models_file=${models_file:-None} system=${system:-linux} step=${step:-train} reponame=${reponame:-PaddleClas} mode=${mode:-function} use_build=${use_build:-yes} branch=${branch:-develop} mode=${mode:-function} get_repo=${get_repo:-clone} paddle_whl=${paddle_whl:-None} dataset_org=${dataset_org:-None} dataset_target=${dataset_target:-None} set_cuda=${set_cuda:-}
    " &
    wait $!
    exit $?
else
    export Project_path=${Project_path:-${PWD}/task/PaddleClas}
    echo ${Project_path}
    ldconfig;
    if [[ ${Python_env} == 'ln_way' ]];then
        rm -rf /usr/bin/python2.7
        rm -rf /usr/local/python2.7.15/bin/python
        rm -rf /usr/local/bin/python
        export PATH=/usr/local/bin/python:${PATH}
        case ${Python_version} in
        36)
        ln -s /usr/local/bin/python3.6 /usr/local/bin/python
        # mkdir run_env_py36;
        # ln -s $(which python3.6) run_env_py36/python;
        # ln -s $(which pip3.6) run_env_py36/pip;
        # export PATH=$(pwd)/run_env_py36:${PATH};
        ;;
        37)
        ln -s /usr/local/bin/python3.7 /usr/local/bin/python
        # mkdir run_env_py37;
        # ln -s $(which python3.7) run_env_py37/python;
        # ln -s $(which pip3.7) run_env_py37/pip;
        # export PATH=$(pwd)/run_env_py37:${PATH};
        ;;
        38)
        ln -s /usr/local/bin/python3.8 /usr/local/bin/python
        # mkdir run_env_py38;
        # ln -s $(which python3.8) run_env_py38/python;
        # ln -s $(which pip3.8) run_env_py38/pip;
        # export PATH=$(pwd)/run_env_py38:${PATH};
        ;;
        39)
        ln -s /usr/local/bin/python3.9 /usr/local/bin/python
        # mkdir run_env_py39;
        # ln -s $(which python3.9) run_env_py39/python;
        # ln -s $(which pip3.9) run_env_py39/pip;
        # export PATH=$(pwd)/run_env_py39:${PATH};
        ;;
        310)
        ln -s /usr/local/bin/python3.10 /usr/local/bin/python
        # mkdir run_env_py310;
        # ln -s $(which python3.10) run_env_py310/python;
        # ln -s $(which pip3.10) run_env_py310/pip;
        # export PATH=$(pwd)/run_env_py310:${PATH};
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
    python main.py models_list=${models_list:-None} models_file=${models_file:-None} system=${system:-linux} step=${step:-train} reponame=${reponame:-PaddleClas} mode=${mode:-function} use_build=${use_build:-yes} branch=${branch:-develop} mode=${mode:-function} get_repo=${get_repo:-clone} paddle_whl=${paddle_whl:-None} dataset_org=${dataset_org:-None} dataset_target=${dataset_target:-None} set_cuda=${set_cuda:-}

fi
