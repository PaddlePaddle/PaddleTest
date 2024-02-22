#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers
echo ${work_path}

log_dir=${root_path}/ut_log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}

cd ${work_path}
exit_code=0

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pip install pytest safetensors ftfy fastcore opencv-python einops parameterized requests-mock
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# export http_proxy=${proxy}
# export https_proxy=${proxy}
# rm -rf tests/pipelines/test_pipelines.py
# rm -rf tests/pipelines/stable_diffusion/test_stable_diffusion_pix2pix_zero.py

exit_code=0

export HF_ENDPOINT=https://hf-mirror.com
export no_proxy=baidu.com,127.0.0.1,0.0.0.0,localhost,bcebos.com,pip.baidu-int.com,mirrors.baidubce.com,repo.baidubce.com,repo.bcm.baidubce.com,pypi.tuna.tsinghua.edu.cn,aistudio.baidu.com
export USE_PPXFORMERS=True
export RUN_SLOW=True
echo "*******ppdiffusers ut tests begin***********"
(python -m pytest -v tests) 2>&1 | tee ${log_dir}/tests_ut.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers ut tests  run success" >>"${log_dir}/ut_res.log"
else
    echo "ppdiffusers ut tests  run fail" >>"${log_dir}/ut_res.log"
fi
echo "*******ppdiffusers ut tests end***********"

# unset http_proxy
# unset https_proxy

# # 查看结果
cat ${log_dir}/ut_res.log

echo exit_code:${exit_code}
exit ${exit_code}
