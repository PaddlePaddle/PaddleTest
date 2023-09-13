#!/bin/bash

cur_path=`pwd`
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/examples
echo ${work_path}

log_dir=${cur_path}/../log

# 检查上一级目录中是否存在log目录
if [ ! -d "$log_dir" ]; then
    # 如果log目录不存在，则创建它
    mkdir -p "$log_dir"
fi

cd ${work_path}/dreambooth
exit_code=0
# 下载依赖、数据集和权重
bash dreambooth_prepare.sh
# lora dreambooth_train
echo "*******lora dreambooth_train begin***********"
bash dreambooth_train.sh.sh > ${log_dir}/lora_dreambooth_train.sh.log 2>&1
exit_code=$(($exit_code + $?))
if [ $? -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "run lora_dreambooth_train run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "run lora_dreambooth_train run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******lora dreambooth_train end***********"

# lora dreambooth_infer
echo "*******lora dreambooth_infer begin***********"
python dreambooth_infer.py > ${log_dir}/lora_dreambooth_infer.log 2>&1
exit_code=$(($exit_code + $?))
# 检查infer.py的返回状态
if [ $? -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "run lora dreambooth_infer run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "run lora dreambooth_infer run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******lora dreambooth_infer end***********"

cd ${work_path}/text_to_image
# 下载依赖
bash text_to_image_prepare.sh
exit_code=$(($exit_code + $?))
echo "*******lora text_to_image train begin***********"
bash text_to_image_train.sh > ${log_dir}/lora_text_to_image_train.log 2>&1
exit_code=$(($exit_code + $?))
if [ $? -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "run lora text_to_image run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "run lora text_to_image run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******lora text_to_image train end***********"

# 多机训练的结果进行推理
echo "*******lora text_to_imageinfer begin***********"
python text_to_image_infer.py> ${log_dir}/lora_text_to_image_train.log 2>&1
exit_code=$(($exit_code + $?))
# 检查infer.py的返回状态
if [ $? -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "run lora text_to_image run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "run lora text_to_image run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******lora text_to_image infer end***********"

echo exit_code:${exit_code}
exit ${exit_code}