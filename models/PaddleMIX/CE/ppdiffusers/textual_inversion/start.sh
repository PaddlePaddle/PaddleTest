#!/bin/bash

cur_path=`pwd`
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/examples/textual_inversion
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

bash prepare.sh
# 单机训练
echo "*******textual_inversion singe_train begin***********"
(bash single_train.sh) 2>&1 | tee ${log_dir}/textual_inversion_singe_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "textual_inversion singe_train run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "textual_inversion singe_train run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******textual_inversion singe_train end***********"

# 单机训练的结果进行推理
echo "******textual_inversion singe infer begin***********"
(python infer_with_output.py 2>&1) | tee ${log_dir}/textual_inversion_single_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "textual_inversion single_infer run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "textual_inversion single_infer run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******textual_inversion singe infer end***********"

# 多机训练
echo "*******textual_inversion muti_train begin***********"
(bash multi_train.sh) 2>&1 | tee ${log_dir}/textual_inversion_multi_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "textual_inversion multi_train run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "textual_inversion multi_train run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******textual_inversion multi_train end***********"

# 多机训练的结果进行推理
echo "*******textual_inversion multi infer begin***********"
(python infer_with_output.py) 2>&1 | tee ${log_dir}/textual_inversion_multi_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "textual_inversion multi_infer run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "textual_inversion multi_infer run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******textual_inversion multi infer end***********"



# # 查看结果
# cat ${log_dir}/ce_res.log

echo exit_code:${exit_code}
exit ${exit_code}