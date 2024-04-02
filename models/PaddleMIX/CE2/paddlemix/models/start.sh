#!/bin/bash

export PPNLP_HOME=/home/cache_weight
export PPMIX_HOME=/home/cache_weight

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/tests/models
echo ${work_path}

log_dir=${root_path}/log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}

cd ${work_path}

exit_code=0

echo "*******paddlemix models test_cviw begin***********"
(python -m pytest -v test_blip2.py) 2>&1 | tee ${log_dir}/test_blip2.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix models test_blip2 success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix models test_blip2 fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix models test_blip2 end***********"

echo "*******paddlemix models test_clip begin***********"
(python -m pytest -v test_clip.py) 2>&1 | tee ${log_dir}/test_clip.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix models test_clip success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix models test_clip fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix models test_clip end***********"

echo "*******paddlemix models test_coca begin***********"
(python -m pytest -v test_coca.py) 2>&1 | tee ${log_dir}/test_coca.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix models test_coca success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix models test_coca fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix models test_coca end***********"

echo "*******paddlemix models test_eva02 begin***********"
(python -m pytest -v test_eva02.py) 2>&1 | tee ${log_dir}/test_eva02.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix models test_eva02 success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix models test_eva02 fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix models test_eva02 end***********"

echo "*******paddlemix models test_evaclip begin***********"
(python -m pytest -v test_evaclip.py) 2>&1 | tee ${log_dir}/test_evaclip.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix models test_evaclip success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix models test_evaclip fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix models test_evaclip end***********"

echo "*******paddlemix models test_minigpt4 begin***********"
(python -m pytest -v test_minigpt4.py) 2>&1 | tee ${log_dir}/test_minigpt4.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix models test_minigpt4 success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix models test_minigpt4 fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix models test_minigpt4 end***********"

echo "*******paddlemix models test_groundingdino begin***********"
(python -m pytest -v test_groundingdino.py) 2>&1 | tee ${log_dir}/test_groundingdino.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix models test_groundingdino success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix models test_groundingdino fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix models test_groundingdino end***********"

echo "*******paddlemix models test_sam begin***********"
(python -m pytest -v test_sam.py) 2>&1 | tee ${log_dir}/test_sam.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix models test_sam success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix models test_sam fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix models test_sam end***********"

echo exit_code:${exit_code}
exit ${exit_code}
