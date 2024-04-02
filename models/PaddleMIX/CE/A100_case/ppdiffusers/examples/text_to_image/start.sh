#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/examples/dreambooth/
echo ${work_path}

log_dir=${root_path}/log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}

cd ${work_path}
exit_code=0

bash prepare.sh


echo "*******text_to_image train begin***********"
(bash train.sh) 2>&1 | tee ${log_dir}/text_to_image_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "text_to_image train run success" >>"${log_dir}/ce_res.log"
else
    echo "text_to_image train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******text_to_image train end***********"


echo "******text_to_image infer begin***********"
(python infer.py)2>&1 | tee ${log_dir}/text_to_image_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "text_to_image infer run success" >>"${log_dir}/ce_res.log"
else
    echo "text_to_image infer run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******text_to_image infer end***********"

echo "*******text_to_image lora_train begin***********"
(bash lora_train.sh) 2>&1 | tee ${log_dir}/text_to_image_lora_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "text_to_image lora_train run success" >>"${log_dir}/ce_res.log"
else
    echo "text_to_image lora_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******text_to_image lora_train end***********"


echo "******text_to_image lora_infer begin***********"
(python lora_infer.py)2>&1 | tee ${log_dir}/text_to_image_lora_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "text_to_image lora_infer run success" >>"${log_dir}/ce_res.log"
else
    echo "text_to_image lora_infer run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******text_to_image lora_infer end***********"

# # 查看结果
# cat ${log_dir}/ce_res.log
rm -rf ${work_path}/sdxl-pokemon-model/*
rm -rf ${work_path}/sd-pokemon-model-lora-sdxl/*

echo exit_code:${exit_code}
exit ${exit_code}
