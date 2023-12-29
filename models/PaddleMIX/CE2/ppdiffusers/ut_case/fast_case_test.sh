#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com

export RUN_SLOW=False
export RUN_NIGHTLY=False

export FROM_DIFFUSERS=True
export TO_DIFFUSERS=True

export FROM_HF_HUB=True
export PATCH_ALLCLOSE=True
export HF_HUB_OFFLINE=False
export CUDA_VISIBLE_DEVICES=0
export no_proxy=su.bcebos.com,hf-mirror.com,baidu.com,127.0.0.1,0.0.0.0,localhost,bj.bcebos.com,pypi.tuna.tsinghua.edu.cn,cdn-lfs.huggingface.co

# 设置已经缓存好的部分文件。
export PPNLP_HOME=/home/weight/ppnlp_home
export HUGGINGFACE_HUB_CACHE=/home/weight/huggingface_home

exit_code=0

export http_proxy=${proxy}
export https_proxy=${proxy}
echo "*******ppdiffusers fast_case_test begin***********"
(python -m pytest -v tests -n 2 -s) 2>&1 | tee ${log_dir}/fast_case_test.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers fast_case_test success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers fast_case_test fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers fast_case_test end***********"
unset http_proxy
unset https_proxy

echo exit_code:${exit_code}
exit ${exit_code}
