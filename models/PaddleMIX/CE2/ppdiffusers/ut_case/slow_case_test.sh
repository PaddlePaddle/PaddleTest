#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com

# 下面2个是开启slow单测的标记,
export RUN_SLOW=True
export RUN_NIGHTLY=True

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

echo "*******ppdiffusers slow_case_test tests/models begin***********"
(python -m pytest -v tests/models -n 2 -s) 2>&1 | tee ${log_dir}/slow_case_test_model.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers slow_case_test tests/models success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers slow_case_test tests/models fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers slow_case_test tests/models end***********"

echo "*******ppdiffusers slow_case_test tests/schedulers begin***********"
(python -m pytest -v tests/schedulers -n 4 -s) 2>&1 | tee ${log_dir}/slow_case_test_schedulers.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers slow_case_test tests/schedulers success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers slow_case_test tests/schedulers fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers slow_case_test tests/schedulers end***********"

echo "*******ppdiffusers slow_case_test tests/others begin***********"
(python -m pytest -v tests/others -n 1 -s) 2>&1 | tee ${log_dir}/slow_case_test_others.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers slow_case_test tests/others success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers slow_case_test tests/others fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers slow_case_test tests/others end***********"

echo "*******ppdiffusers slow_case_test tests/pipelines begin***********"
(python -m pytest -v tests/pipelines -n 1 -s) 2>&1 | tee ${log_dir}/slow_case_test_pipelines.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers slow_case_test tests/pipelines success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers slow_case_test tests/pipelines fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers slow_case_test tests/pipelines end***********"

unset http_proxy
unset https_proxy

echo exit_code:${exit_code}
exit ${exit_code}
