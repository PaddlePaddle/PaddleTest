#!/usr/bin/env bash
# 0415
# param:
#imagename= paddlepaddle/paddle_manylinux_devel:cuda10.1-cudnn7
#paddle= develop_0.0.0\release_1.8.3
#paddleslim = slim1_install\slim2_build
#python= 36
#################################
python -m pip install PyGithub
git config --global user.name "Paddle CI"
git config --global user.email "paddle_ci@example.com"
git remote | grep upstream
if [ $? == 1 ]; then git remote add upstream https://github.com/PaddlePaddle/PaddleSlim.git; fi
set -e
git fetch upstream
git checkout -b origin_pr
git checkout -b test_pr upstream/%BRANCH%
git merge origin_pr
git log --pretty=oneline -10
set +e
#################################
set -e
export GIT_PR_ID=
branch=$(echo "%teamcity.build.branch%")
echo $branch
if [[ $branch == *"pull/"* ]]; then
    export GIT_PR_ID=$(echo $branch | sed 's/[^0-9]*//g')
fi
echo "GIT_PR_ID: ""${GIT_PR_ID}"
mkdir ci_approve
cd ci_approve
wget -q https://sys-p0.bj.bcebos.com/models_ci/approve_tools.tgz --no-check-certificate
tar -zxf approve_tools.tgz
wget -q https://paddle-ci.gz.bcebos.com/blk/block.txt --no-check-certificate
wget -q https://sys-p0.bj.bcebos.com/bk-ci/bk.txt --no-check-certificate
sh tools/check_file_diff_approvals.sh PaddleSlim
cd -
set +e
#################################
export P0case_list=()
export P0case_time=0
export all_P0case_time=0
declare -A all_P0case_dic
all_P0case_dic=(["distillation"]=5 ["quant"]=15 ["prune"]=15 ["nas"]=30 ["darts"]=30)
get_diff_TO_P0case(){
for key in $(echo ${!all_P0case_dic[*]});do
    all_P0case_time=`expr ${all_P0case_time} + ${all_P0case_dic[$key]}`
done
for file_name in `git diff --numstat upstream/develop |awk '{print $NF}'`;do
    arr_file_name=(${file_name//// })
    dir1=${arr_file_name[0]}
    dir2=${arr_file_name[1]}
    echo "file_name:"$file_name, "dir1:"$dir1, "dir2:"$dir2
    if [[ ${file_name##*.} =~ "md" ]] || [[ ${dir1} =~ "docs" ]] || [[ ${file_name##*.} =~ "rst" ]] || [[ ${dir1} =~ "tests" ]] || [[ ${file_name##*.} =~ "jpg" ]] || [[ ${file_name##*.} =~ "png" ]] ;then
        continue
    elif [[ ${dir1} =~ "demo" ]] && [[ ${!all_P0case_dic[*]} =~ ${dir2} ]];then
        P0case_list[${#P0case_list[*]}]=${dir2}
        P0case_time=`expr ${P0case_time} + ${all_P0case_dic[${dir2}]}`
    else
        P0case_list=(distillation quant prune nas darts)
        P0case_time=${all_P0case_time}
        break
    fi
done
}
set -e
get_diff_TO_P0case
echo -e "\033[35m ---- P0case_list length: ${#P0case_list[*]}, cases: ${P0case_list[*]} \033[0m"
echo -e "\033[35m ---- P0case_time: $P0case_time min \033[0m"
set +e
if  [[ ${#P0case_list[*]} == 0 ]];then
echo -e "\033[32m skip p0case \033[0m";
exit 0;
fi
#################################
#git clone script
#################################
#依据agent设置所使用的GPU卡数,此处修改后多卡的启动脚本中无需再设置 --gpus='0,1'
tc_name=`(echo $PWD|awk -F '/' '{print $3}')`
echo "teamcity path:" $tc_name
if [ $tc_name == "teamcity1" ];then
   cudaid1=0;
   cudaid2=0,1;
elif [ $tc_name == "teamcity2" ];then
   cudaid1=2;
   cudaid2=2,3;
elif [ $tc_name == "teamcity3" ];then
   cudaid1=4;
   cudaid2=4,5;
elif [ $tc_name == "teamcity4" ];then
   cudaid1=6;
   cudaid2=6,7;
fi
#################################
export CUDA_SO="$(\ls /usr/lib64/libcuda* | xargs -I{} echo '-v {}:{}') $(\ls /usr/lib64/libnvidia* | xargs -I{} echo '-v {}:{}')"
export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
export NVIDIA_SMI="-v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi"
export GIT_PATH="%system.agent.home.dir%"/system/git;

docker pull %image_name% ;
nvidia-docker run -i --rm $CUDA_SO $DEVICES $NVIDIA_SMI \
--name Slim_CI_${GIT_PR_ID}_$RANDOM --privileged \
--security-opt seccomp=unconfined --net=host \
--shm-size=50G \
-v $PWD:/workspace  \
-v /ssd1/guomengmeng01:/paddle \
-v /ssd1:/ssd1 \
-v ${GIT_PATH}:${GIT_PATH} \
-w /workspace \
-e "GIT_PR_ID=${GIT_PR_ID}" \
-e "GITHUB_API_TOKEN=${GITHUB_API_TOKEN}" \
-e "P0case_list=${P0case_list}" \
-e "cudaid1=${cudaid1}" \
-e "cudaid2=${cudaid2}" \
 %image_name% \
/bin/bash -c "
set -x
export http_proxy=%http_proxy%;
export https_proxy=%http_proxy%;
cp -r /paddle/slim_ci/. ./;
bash slim_ci.sh %python% %paddle% %paddleSlim% %http_proxy% ${cudaid1} ${cudaid2};
"
exit_code=$?
if [ $exit_code != 0 ]
then
    echo "FAIL"
    exit $exit_code
fi

