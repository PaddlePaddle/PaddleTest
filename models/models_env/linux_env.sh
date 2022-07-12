set +x;
pwd;

####ce框架根目录
rm -rf ce && mkdir ce;
cd ce;

####测试框架下载
"""
通过CE_version判断使用V1还是V2版本
"""
if [[ ${CE_version}=="V1" ]];then
    CE_version_name=${CE_version_name}
    wget -q ${CE_V1}
else
    CE_version_name=continuous_evaluation
    wget -q ${CE_V2}
fi
unzip -P ${CE_pass}  ${CE_version_name}.zip

####设置代理  proxy不单独配置 表示默认有全部配置，不用export
if  [ ! -n "${proxy}" ] ;then
    echo unset http_proxy
else
    export http_proxy=${proxy}
    export https_proxy=${proxy}
fi
export no_proxy=${no_proxy}
set -x;
ls;

####之前下载过了直接mv
mv ../PaddleTest .

#通用变量[用户改]
test_code_download_path=./PaddleTest/models/${Repo}/CE
test_code_download_path_CI=./PaddleTest/models/${Repo}/CI
test_code_conf_path=./PaddleTest/models/${Repo}/CE/conf  #各个repo自己管理，可以分类，根据任务类型copy对应的common配置

#迁移下载路径代码和配置到框架指定执行路径 [不用改]
mkdir -p ${test_code_download_path}/log
ls ${test_code_download_path}/log;
cp -r ${test_code_download_path}/.  ./${CE_version_name}/src/PaddleTest
cp -r ${test_code_download_path_CI}/.  ./${CE_version_name}/src/PaddleTest
cp ${test_code_conf_path}/${Common_name}.py ./${CE_version_name}/src/PaddleTest/common.py
cat ./${CE_version_name}/src/PaddleTest/common.py;
echo "环境准备后的目录结构：";
ls;

####根据agent制定对应卡，记得起agent时文件夹按照release_01 02 03 04名称
tc_name=`(echo $PWD|awk -F '/' '{print $4}')`
echo "teamcity path:" $tc_name
if [ $tc_name == "release_02" ];then
    echo release_02
    sed -i "s/SET_CUDA = \"0\"/SET_CUDA = \"2\"/g"  ./${CE_version_name}/src/PaddleTest/common.py
    sed -i "s/SET_MULTI_CUDA = \"0,1\"/SET_MULTI_CUDA = \"2,3\"/g" ./${CE_version_name}/src/PaddleTest/common.py
    SET_CUDA=2;
    SET_MULTI_CUDA=2,3;

elif [ $tc_name == "release_03" ];then
    echo release_03
    sed -i "s/SET_CUDA = \"0\"/SET_CUDA = \"4\"/g"  ./${CE_version_name}/src/PaddleTest/common.py
    sed -i "s/SET_MULTI_CUDA = \"0,1\"/SET_MULTI_CUDA = \"4,5\"/g" ./${CE_version_name}/src/PaddleTest/common.py
    SET_CUDA=4;
    SET_MULTI_CUDA=4,5;

elif [ $tc_name == "release_04" ];then
    echo release_04
    sed -i "s/SET_CUDA = \"0\"/SET_CUDA = \"6\"/g"  ./${CE_version_name}/src/PaddleTest/common.py
    sed -i "s/SET_MULTI_CUDA = \"0,1\"/SET_MULTI_CUDA = \"6,7\"/g"  ./${CE_version_name}/src/PaddleTest/common.py
    SET_CUDA=6;
    SET_MULTI_CUDA=6,7;
else
    echo release_01
    SET_CUDA=0;
    SET_MULTI_CUDA=0,1;

fi

####显示执行步骤
cat ./${CE_version_name}/src/PaddleTest/common.py

#####进入执行路径创建docker容器 [用户改docker创建]
cd ${CE_version_name}/src

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
nvidia-docker run -i   --rm \
             --name=${docker_name} --net=host \
             --shm-size=128G \
             -v $(pwd):/workspace \
             -v /ssd2:/ssd2 \
             -w /workspace \
            ${Image_version}  \
            /bin/bash -c "

            export no_proxy=${no_proxy}
            if [[ ${proxy}=='None']];then
                echo unset http_proxy
            else
                export http_proxy=${http_proxy}
                export https_proxy=${http_proxy}
            fi
            if  [ ! -n '${model_flag}' ] ;then
                echo 'you have not input a model_flag!'
            else
                export model_flag=${model_flag}
            fi
            export Data_path=${Data_path}
            export Project_path=${Project_path}
            export SET_CUDA=${SET_CUDA}
            export SET_MULTI_CUDA=${SET_MULTI_CUDA}
             
            ldconfig;
            echo python_version;
            if [[ ${Python_env}=='path_way']];then
                case ${Python_version} in 
                36)
                export LD_LIBRARY_PATH=/opt/_internal/cpython-3.6.0/lib/:/usr/local/ssl/lib:/opt/rh/devtoolset-2/root/usr/lib64:/opt/rh/devtoolset-2/root/usr/lib:/usr/local/lib64:/usr/local/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64;
                export PATH=/opt/_internal/cpython-3.6.0/bin/:/usr/local/ssl:/usr/local/go/bin:/root/gopath/bin:/usr/local/gcc-8.2/bin:/opt/rh/devtoolset-2/root/usr/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin;
                ;;
                37)
                export LD_LIBRARY_PATH=/opt/_internal/cpython-3.7.0/lib/:/usr/local/ssl/lib:/opt/rh/devtoolset-2/root/usr/lib64:/opt/rh/devtoolset-2/root/usr/lib:/usr/local/lib64:/usr/local/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64;
                export PATH=/opt/_internal/cpython-3.7.0/bin/:/usr/local/ssl:/usr/local/go/bin:/root/gopath/bin:/usr/local/gcc-8.2/bin:/opt/rh/devtoolset-2/root/usr/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin;
                ;;
                38)
                export LD_LIBRARY_PATH=/opt/_internal/cpython-3.8.0/lib/:/usr/local/ssl/lib:/opt/rh/devtoolset-2/root/usr/lib64:/opt/rh/devtoolset-2/root/usr/lib:/usr/local/lib64:/usr/local/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64;
                export PATH=/opt/_internal/cpython-3.8.0/bin/:/usr/local/ssl:/usr/local/go/bin:/root/gopath/bin:/usr/local/gcc-8.2/bin:/opt/rh/devtoolset-2/root/usr/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin;
                ;;
                39)
                export LD_LIBRARY_PATH=/opt/_internal/cpython-3.9.0/lib/:/usr/local/ssl/lib:/opt/rh/devtoolset-2/root/usr/lib64:/opt/rh/devtoolset-2/root/usr/lib:/usr/local/lib64:/usr/local/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64;
                export PATH=/opt/_internal/cpython-3.9.0/bin/:/usr/local/ssl:/usr/local/go/bin:/root/gopath/bin:/usr/local/gcc-8.2/bin:/opt/rh/devtoolset-2/root/usr/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin;
                ;;
                esac
            elif [[ ${Python_env}=='ln_way']];then
                case ${python} in #python
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
                echo unset python version
            fi

            python -c 'import sys; print(sys.version_info[:])';
            git --version;

            bash main.sh --PaddleTest_type='model' --build_number=${AGILE_PIPELINE_BUILD_NUMBER} --project_name=${AGILE_MODULE_NAME} --PaddleTest_name=${AGILE_PIPELINE_NAME}  --build_id=${AGILE_PIPELINE_BUILD_ID} --build_type=${AGILE_PIPELINE_UUID}  --owner='liuquanxiang' --priority=${Priority_version} --compile_path=${Compile_version}  --agile_job_build_id=${AGILE_JOB_BUILD_ID}
  " &
wait $!
exit $?
