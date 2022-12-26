set +x;
pwd;

####ce框架根目录
rm -rf ce && mkdir ce;
cd ce;

# 使虚拟环境生效
source ~/.bashrc

######################## 定义变量 ########################
# AGILE_PIPELINE_NAME 格式类似: PaddleClas-MAC-Intel-Python310-P9-Develop
#其它内容或者可能不一致的不要随意加 "-", 下面是按照 "-" split 按序号填入的

#repo的名称
export reponame=${reponame:-"`(echo ${AGILE_PIPELINE_NAME}|awk -F '-' '{print $1}')`"}

#模型列表文件 , 固定路径及格式为 tools/reponame_优先级_list   优先级P2有多个用P21、P22  中间不用"-"划分, 防止按 "-" split 混淆
export models_file=${models_file:-"tools/${reponame}_`(echo ${AGILE_PIPELINE_NAME}|awk -F '-' '{print $5}')`_list"}
export models_list=${models_list:-None} #模型列表

#指定case操作系统
if [[ ${AGILE_PIPELINE_NAME} =~ "Linux" ]];then
    export system=${system:-"linux"}   # linux windows windows_cpu mac 与yaml case下字段保持一致
elif [[ ${AGILE_PIPELINE_NAME} =~ "Windows" ]];then
    export system=${system:-"windows"}
elif [[ ${AGILE_PIPELINE_NAME} =~ "WindowsCPU" ]];then
    export system=${system:-"windows_cpu"}
elif [[ ${AGILE_PIPELINE_NAME} =~ "Mac" ]] ;then
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
if [[ ${Python_version} =~ "36" ]];then
    pyenv activate ${reponame}_36
elif [[ ${Python_version} =~ "37" ]];then
    pyenv activate ${reponame}_37
elif [[ ${Python_version} =~ "38" ]];then
    pyenv activate ${reponame}_38
elif [[ ${Python_version} =~ "39" ]];then
    pyenv activate ${reponame}_39
elif [[ ${Python_version} =~ "310" ]];then
    pyenv activate ${reponame}_310
else
    pyenv activate ${reponame}_310
    echo "default set python verison is python3.10"
fi

if [[ ${AGILE_PIPELINE_NAME} =~ "-Intel-" ]];then
    if [[ ${AGILE_PIPELINE_NAME} =~ "Develop" ]];then
        export paddle_whl=${paddle_whl:-"https://paddle-wheel.bj.bcebos.com/2.1.2/macos/macos-cpu-openblas/paddlepaddle-0.0.0-cp310-cp310-macosx_10_14_universal2.whl"}
    else
        export paddle_whl=${paddle_whl:-"https://paddle-wheel.bj.bcebos.com/2.1.2/macos/macos-cpu-openblas/paddlepaddle-0.0.0-cp310-cp310-macosx_10_14_universal2.whl"}
    fi
elif [[ ${AGILE_PIPELINE_NAME} =~ "-M1-" ]];then
    if [[ ${AGILE_PIPELINE_NAME} =~ "Develop" ]];then
        export paddle_whl=${paddle_whl:-"https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-Build-Mac-M1/latest/paddlepaddle-0.0.0-cp310-cp310-macosx_11_0_arm64.whl"}
    else
        export paddle_whl=${paddle_whl:-"https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-Cpu-Mac-Arm-Py310-Compile/latest/paddlepaddle-0.0.0-cp310-cp310-macosx_11_0_arm64.whl"}
    fi
fi
#### 预设默认参数
export step=${step:-train}
export mode=${mode:-function}
export use_build=${use_build:-yes}
export branch=${branch:-develop}
export get_repo=${get_repo:-wget} #现支持10个库，需要的话可以加，wget快很多
export dataset_org=${dataset_org:-None}
export dataset_target=${dataset_target:-None}

#额外的变量
export http_proxy=${http_proxy:-}
export no_proxy=${no_proxy:-}

# 预设一些可能会修改的变量
export CE_version_name=${CE_version_name:-TestFrameWork}
export models_name=${models_name:-models_restruct}

####测试框架下载
wget -q ${CE_Link} #需要全局定义
unzip -P ${CE_pass} ${CE_version_name}.zip

####设置代理  proxy不单独配置 表示默认有全部配置，不用export
if  [[ ${http_proxy} ]] ;then
    export http_proxy=${http_proxy}
    export https_proxy=${http_proxy}
else
    echo "unset http_proxy"
fi
export no_proxy=${no_proxy}
set -x;

#输出参数验证
echo "@@@reponame: ${reponame}"
echo "@@@models_list: ${models_list}"
echo "@@@models_file: ${models_file}"
echo "@@@system: ${system}"
echo "@@@Python_version: ${Python_version}"
echo "@@@paddle_whl: ${paddle_whl}"
echo "@@@step: ${step}"
echo "@@@branch: ${branch}"
echo "@@@mode: ${mode}"

####之前下载过了直接mv
if [[ -d "../task" ]];then
    mv ../task .  #如果预先下载直接mv
else
    wget -q https://xly-devops.bj.bcebos.com/PaddleTest/PaddleTest.tar.gz --no-proxy  >/dev/null
    tar xf PaddleTest.tar.gz >/dev/null 2>&1
    mv PaddleTest task
fi

##如果预先模型库下载直接mv, 方便二分是checkout 到某个commit进行二分
if [[ -d "../../${reponame}" ]];then  #前面cd 了 2次所以使用 ../../
    cp -r ../../${reponame} .
    echo "因为 ${reponame} 在根目录存在 使用预先clone或wget的 ${reponame}"
fi

#复制模型相关文件到指定位置
cp -r ./task/${models_name}/${reponame}/.  ./${CE_version_name}/
ls ./${CE_version_name}/
cd ./${CE_version_name}/

python -c 'import sys; print(sys.version_info[:])';
git --version;
python -m pip install -U pip #升级pip
python -m pip install -r requirements.txt #预先安装依赖包
python main.py --models_list=${models_list:-None} --models_file=${models_file:-None} --system=${system:-linux} --step=${step:-train} --reponame=${reponame:-PaddleClas} --mode=${mode:-function} --use_build=${use_build:-yes} --branch=${branch:-develop} --get_repo=${get_repo:-wget} --paddle_whl=${paddle_whl:-None} --dataset_org=${dataset_org:-None} --dataset_target=${dataset_target:-None}
