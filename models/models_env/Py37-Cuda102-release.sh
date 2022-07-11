#exit 0;
set +x;
pwd;
rm -rf ce && mkdir ce;
cd ce;
#--测试框架下载
set +x;
#下载v2 CE测试框架
wget -O Paddle_Cloud_CE.zip -q  ${CE_framwork}
# 解压
set +x;
unzip -P ${CE_password}  Paddle_Cloud_CE.zip
if [[ -d "continuous_evaluation" ]];then
    mv continuous_evaluation Paddle_Cloud_CE
fi
#  设置代理
export http_proxy=${http_proxy}
export https_proxy=${https_proxy}
export no_proxy=${no_proxy}
set -x;
ls;

mv ../task .

#通用变量[用户改]
test_code_download_path=./task/models/PaddleClas/CE
test_code_download_path_CI=./task/models/PaddleClas/CI
test_code_conf_path=./task/models/PaddleClas/CE/conf  #各个repo自己管理，可以分类，根据任务类型copy对应的common配置

#迁移下载路径代码和配置到框架指定执行路径 [不用改]
mkdir -p ${test_code_download_path}/log
ls ${test_code_download_path}/log;
cp -r ${test_code_download_path}/.  ./Paddle_Cloud_CE/src/task
cp -r ${test_code_download_path_CI}/.  ./Paddle_Cloud_CE/src/task
cp ${test_code_conf_path}/cls_common_release.py ./Paddle_Cloud_CE/src/task/common.py
cat ./Paddle_Cloud_CE/src/task/common.py;
echo "环境准备后的目录结构：";
ls;

##############
tc_name=`(echo $PWD|awk -F '/' '{print $4}')`
echo "teamcity path:" $tc_name
if [ $tc_name == "release_02" ];then
    echo release_02
    sed -i "s/SET_CUDA = \"0\"/SET_CUDA = \"2\"/g"  ./Paddle_Cloud_CE/src/task/common.py
    sed -i "s/SET_MULTI_CUDA = \"0,1\"/SET_MULTI_CUDA = \"2,3\"/g" ./Paddle_Cloud_CE/src/task/common.py
    SET_CUDA=2;
    SET_MULTI_CUDA=2,3;

elif [ $tc_name == "release_03" ];then
    echo release_03
    sed -i "s/SET_CUDA = \"0\"/SET_CUDA = \"4\"/g"  ./Paddle_Cloud_CE/src/task/common.py
    sed -i "s/SET_MULTI_CUDA = \"0,1\"/SET_MULTI_CUDA = \"4,5\"/g" ./Paddle_Cloud_CE/src/task/common.py
    SET_CUDA=4;
    SET_MULTI_CUDA=4,5;

elif [ $tc_name == "release_04" ];then
    echo release_04
    sed -i "s/SET_CUDA = \"0\"/SET_CUDA = \"6\"/g"  ./Paddle_Cloud_CE/src/task/common.py
    sed -i "s/SET_MULTI_CUDA = \"0,1\"/SET_MULTI_CUDA = \"6,7\"/g"  ./Paddle_Cloud_CE/src/task/common.py
    SET_CUDA=6;
    SET_MULTI_CUDA=6,7;
else
    echo release_01
    SET_CUDA=0;
    SET_MULTI_CUDA=0,1;

fi
cat ./Paddle_Cloud_CE/src/task/common.py
##############
#进入执行路径创建docker容器 [用户改docker创建]
cd Paddle_Cloud_CE/src

set +x;
docker_name="ce_clas_p0_release_${AGILE_PIPELINE_BUILD_NUMBER}"
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
             -e "no_proxy=${no_proxy}" \
             -e "http_proxy=${http_proxy}" \
             -e "https_proxy=${http_proxy}" \
             -e "paddle_compile=${paddle_compile}" \
             -e "Data_path=${Data_path_release}" \
             -e "model_flag=${model_flag}" \
             -e "Project_path=${Project_path_release}" \
             -e "cudaid1=${SET_CUDA}" \
             -e "cudaid2=${SET_MULTI_CUDA}" \
             -w /workspace \
            ${image_name}  \
            /bin/bash -c "
            ldconfig;
            echo python_version;
            ls /opt/_internal/;
            export LD_LIBRARY_PATH=/opt/_internal/cpython-3.7.0/lib/:/usr/local/ssl/lib:/opt/rh/devtoolset-2/root/usr/lib64:/opt/rh/devtoolset-2/root/usr/lib:/usr/local/lib64:/usr/local/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64;
            export PATH=/opt/_internal/cpython-3.7.0/bin/:/usr/local/ssl:/usr/local/go/bin:/root/gopath/bin:/usr/local/gcc-8.2/bin:/opt/rh/devtoolset-2/root/usr/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin;
            python -c 'import sys; print(sys.version_info[:])';
            git --version;
            python -m pip install 'numpy<=1.19.3';

            bash main.sh --task_type='model' --build_number=${AGILE_PIPELINE_BUILD_NUMBER} --project_name=${AGILE_MODULE_NAME} --task_name=${AGILE_PIPELINE_NAME}  --build_id=${AGILE_PIPELINE_BUILD_ID} --build_type=${AGILE_PIPELINE_UUID}  --owner='liuquanxiang' --priority=${priority_release} --compile_path=${paddle_compile_release}  --agile_job_build_id=${AGILE_JOB_BUILD_ID}
  " &
wait $!
exit $?
