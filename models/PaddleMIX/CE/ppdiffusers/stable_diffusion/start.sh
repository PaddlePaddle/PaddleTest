#!/bin/bash

cur_path=`pwd`
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/examples/stable_diffusion
echo ${work_path}

log_dir=${root_path}/log

# 检查上一级目录中是否存在log目录
if [ ! -d "$log_dir" ]; then
    # 如果log目录不存在，则创建它
    mkdir -p "$log_dir"
fi


/bin/cp -rf ./* ${work_path}

cd ${work_path}
exit_code=0

# 下载依赖、数据集和权重
bash prepare.sh
# 单机训练
echo "*******stable_diffusion singe_train begin***********"
(bash singe_train.sh) 2>&1 | tee ${log_dir}/stable_diffusion_singe_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "stable_diffusion_singe_train run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "stable_diffusion_singe_train run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******stable_diffusion singe_train end***********"

# 单机训练的结果进行推理
echo "******stable_diffusion singe infer begin***********"
rm -rf astronaut_rides_horse.png
(python infer.py 2>&1) | tee ${log_dir}/stable_diffusion_single_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "stable_diffusion_single_infer run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "stable_diffusion_single_infer run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******stable_diffusion singe infer end***********"

# 多机训练
echo "*******stable_diffusion muti_train begin***********"
(bash muti_train.sh) 2>&1 | tee ${log_dir}/stable_diffusion_muti_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "stable_diffusion_muti_train run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "stable_diffusion_muti_train run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******stable_diffusion muti_train end***********"

# 多机训练的结果进行推理
echo "*******stable_diffusion multi infer begin***********"
rm -rf astronaut_rides_horse.png
(python infer.py) 2>&1 | tee ${log_dir}/stable_diffusion_muti_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "stable_diffusion_multi_infer run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "stable_diffusion_multi_infer run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******stable_diffusion multi infer end***********"

# # 查看结果
# cat ${log_dir}/ce_res.log

echo exit_code:${exit_code}
exit ${exit_code}