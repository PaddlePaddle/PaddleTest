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
export Python_env=${Python_env:-path_way}
#paddlepaddle(ln_way)，manylinux(path_way)
export Python_version=${Python_version:-37}
export Image_version=${Image_version:-registry.baidubce.com/paddlepaddle/paddle_manylinux_devel:cuda10.2-cudnn7}

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

python -c 'import sys; print(sys.version_info[:])';
git --version;
python -m pip install -r requirements.txt #预先安装依赖包
python main.py --models_list=${models_list:-None} --models_file=${models_file:-None} --system=${system:-linux} --step=${step:-train} --reponame=${reponame:-PaddleClas} --mode=${mode:-function} --use_build=${use_build:-yes} --branch=${branch:-develop} --get_repo=${get_repo:-wget} --paddle_whl=${paddle_whl:-None} --dataset_org=${dataset_org:-None} --dataset_target=${dataset_target:-None}
