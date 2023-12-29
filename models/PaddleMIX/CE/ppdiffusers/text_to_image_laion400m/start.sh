#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/examples/text_to_image_laion400m/
echo ${work_path}

log_dir=${root_path}/log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}

cd ${work_path}
exit_code=0

bash prepare.sh
# 单机训练
echo "*******text_to_image_laion400m single_train begin***********"
(bash single_train.sh) 2>&1 | tee ${log_dir}/text_to_image_laion400m_single_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "text_to_image_laion400m single_train run success" >>"${log_dir}/ce_res.log"
else
    echo "text_to_image_laion400m single_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******text_to_image_laion400m single_train end***********"

# 单机训练的结果进行推理
echo "*******text_to_image_laion400m single_infer begin***********"
(bash infer.sh) 2>&1 | tee ${log_dir}/text_to_image_laion400m_single_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "text_to_image_laion400m single_infer run success" >>"${log_dir}/ce_res.log"
else
    echo "text_to_image_laion400m single_infer run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******text_to_image_laion400m single_infer end***********"

# 多机训练
echo "*******text_to_image_laion400m muti_train begin***********"
(bash multi_train.sh) 2>&1 | tee ${log_dir}/text_to_image_laion400m_multi_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "text_to_image_laion400m multi_train run success" >>"${log_dir}/ce_res.log"
else
    echo "text_to_image_laion400m multi_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******text_to_image_laion400m multi_train end***********"

# 多机训练的结果进行推理
echo "*******text_to_image_laion400m multi_infer begin***********"
(bash infer.sh) 2>&1 | tee ${log_dir}/text_to_image_laion400m_multi_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "text_to_image_laion400m multi_infer run success" >>"${log_dir}/ce_res.log"
else
    echo "text_to_image_laion400m multi_infer run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******text_to_image_laion400m multi_infer end***********"

# 自定义训练逻辑开启训练
echo "*******text_to_image_laion400m single_train_no_trainer begin***********"
(bash single_train_no_trainer.sh) 2>&1 | tee ${log_dir}/text_to_image_laion400m_single_train_no_trainer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "text_to_image_laion400m single_train_no_trainer run success" >>"${log_dir}/ce_res.log"
else
    echo "text_to_image_laion400m single_train_no_trainer run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******text_to_image_laion400m single_train_no_trainer end***********"

echo "*******text_to_image_laion400m single infer_no_trainer begin***********"
(bash infer_no_trainer.sh) 2>&1 | tee ${log_dir}/text_to_image_laion400m_single_train_no_trainer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "text_to_image_laion400m single infer_no_trainer run success" >>"${log_dir}/ce_res.log"
else
    echo "text_to_image_laion400m single infer_no_trainer run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******text_to_image_laion400m single infer_no_trainer end***********"

echo "*******text_to_image_laion400m multi_train_no_trainer begin***********"
(bash multi_train_no_trainer.sh) 2>&1 | tee ${log_dir}/text_to_image_laion400m_multi_train_no_trainer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "text_to_image_laion400m multi_train_no_trainer run success" >>"${log_dir}/ce_res.log"
else
    echo "text_to_image_laion400m multi_train_no_trainer run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******text_to_image_laion400m multi_train_no_trainer end***********"

echo "*******text_to_image_laion400m multi infer_no_trainer begin***********"
(bash infer_no_trainer.sh) 2>&1 | tee ${log_dir}/text_to_image_laion400m_multi_train_no_trainer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "text_to_image_laion400m multi infer_no_trainer run success" >>"${log_dir}/ce_res.log"
else
    echo "text_to_image_laion400m multi infer_no_trainer run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******text_to_image_laion400m multi infer_no_trainer end***********"

echo "*******text_to_image_laion400m infer_mscoco begin***********"
(bash infer_mscoco.sh) 2>&1 | tee ${log_dir}/text_to_image_laion400m_infer_mscoco.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "text_to_image_laion400m infer_mscoco run success" >>"${log_dir}/ce_res.log"
else
    echo "text_to_image_laion400m infer_mscoco run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******text_to_image_laion400m infer_mscoco end***********"

# # 查看结果
# cat ${log_dir}/ce_res.log

echo exit_code:${exit_code}
exit ${exit_code}
