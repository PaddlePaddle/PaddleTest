#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/examples/
echo ${work_path}

log_dir=${root_path}/log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}/dreambooth/
cd ${work_path}/dreambooth/
exit_code=0
# 下载依赖、数据集和权重
bash dreambooth_prepare.sh
# lora dreambooth_train
echo "*******lora dreambooth_train begin***********"
(bash dreambooth_train.sh) 2>&1 | tee ${log_dir}/lora_dreambooth_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "lora_dreambooth_train run success" >>"${log_dir}/ce_res.log"
else
    echo "lora_dreambooth_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******lora dreambooth_train end***********"

# lora dreambooth_infer
echo "*******lora dreambooth infer begin***********"
(python dreambooth_infer.py) 2>&1 | tee ${log_dir}/lora_dreambooth_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
# 检查infer.py的返回状态
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "lora dreambooth_infer run success" >>"${log_dir}/ce_res.log"
else
    echo "lora dreambooth_infer run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******lora dreambooth_infer end***********"

/bin/cp -rf ./* ${work_path}/text_to_image/
cd ${work_path}/text_to_image/
# 下载依赖
bash text_to_image_prepare.sh
echo "*******lora text_to_image train begin***********"
(bash text_to_image_train.sh) 2>&1 | tee ${log_dir}/lora_text_to_image_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "lora text_to_image train run success" >>"${log_dir}/ce_res.log"
else
    echo "lora text_to_image train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******lora text_to_image train end***********"

# lora_text_to_image
echo "*******lora text_to_image infer begin***********"
(python text_to_image_infer.py) 2>&1 | tee ${log_dir}/lora_text_to_image_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "lora text_to_image infer run success" >>"${log_dir}/ce_res.log"
else
    echo "lora text_to_image infer run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******lora text_to_image infer end***********"

# # 查看结果
# cat ${log_dir}/lora_ce_res.log

echo exit_code:${exit_code}
exit ${exit_code}
