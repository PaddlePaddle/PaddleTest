#!/bin/bash

cur_path=`pwd`
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/examples/dreambooth/
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
echo "*******dreambooth singe_train begin***********"
(bash single_train.sh) 2>&1 | tee ${log_dir}/dreambooth_singe_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "dreambooth singe_train run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "dreambooth singe_train run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******dreambooth singe_train end***********"

# 单机训练的结果进行推理
echo "******dreambooth singe infer begin***********"
(python infer.py 2>&1) | tee ${log_dir}/dreambooth_single_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "dreambooth single_infer run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "dreambooth single_infer run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******dreambooth singe infer end***********"

# 多机训练
echo "*******dreambooth muti_train begin***********"
(bash multi_train.sh) 2>&1 | tee ${log_dir}/dreambooth_multi_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "dreambooth multi_train run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "dreambooth multi_train run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******dreambooth multi_train end***********"

# 多机训练的结果进行推理
echo "*******dreambooth multi infer begin***********"
(python infer.py) 2>&1 | tee ${log_dir}/dreambooth_multi_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "dreambooth multi_infer run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "dreambooth multi_infer run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******dreambooth multi infer end***********"

# 给模型引入先验知识（图片）一同训练
echo "*******dreambooth train_with_class begin***********"
(python train_with_class.sh) 2>&1 | tee ${log_dir}/dreambooth_train_with_class.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "dreambooth train_with_class run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "dreambooth train_with_class run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******dreambooth train_with_class end***********"

# 给模型引入先验知识（图片）一同训练的结果进行推理
echo "*******dreambooth infer_with_class begin***********"
(python infer_with_class.py) 2>&1 | tee ${log_dir}/dreambooth_infer_with_class.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "dreambooth infer_with_class success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "dreambooth infer_with_class fail" >> "${log_dir}/ce_res.log"
fi
echo "*******dreambooth infer_with_class end***********"


# # 查看结果
# cat ${log_dir}/ce_res.log

echo exit_code:${exit_code}
exit ${exit_code}