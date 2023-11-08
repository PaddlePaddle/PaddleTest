#!/bin/bash

export no_proxy=localhost,bj.bcebos.com,su.bcebos.com;
rm -rf run_env_py310;
mkdir run_env_py310;
ln -s $(which python3.10) run_env_py310/python;
ln -s $(which pip3.10) run_env_py310/pip;
export PATH=$(pwd)/run_env_py310:${PATH};
python --version

export http_proxy=${proxy};
export https_proxy=${proxy};
bash prepare.sh
unset http_proxy
unset https_proxy

export RUN_SLOW=False
export RUN_NIGHTLY=False
export FROM_HF_HUB=False
export FROM_DIFFUSERS=False
export TO_DIFFUSERS=False
export HF_HUB_OFFLINE=False
export CUDA_VISIBLE_DEVICES=$cudaid1

work_path=`pwd`
echo ${work_path}

exit_code=0

cd ${work_path}

cd ${work_path}/paddlemix/
bash fire.sh
exit_code=$(($exit_code + $?))

cd ${work_path}/ppdiffusers/
bash fire.sh
exit_code=$(($exit_code + $?))

cd ${work_path}/ut/
bash fire.sh
exit_code=$(($exit_code + $?))

log_dir=${root_path}/log
cat ${log_dir}/res.log

echo exit_code:${exit_code}
exit ${exit_code}
exit ${exit_code}