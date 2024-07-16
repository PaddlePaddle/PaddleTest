#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/examples/Open-Sora/
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
echo "*******open_sora train begin***********"
(bash train.sh) 2>&1 | tee ${log_dir}/open_sora_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "open_sora_train run success" >>"${log_dir}/ce_res.log"
else
    echo "open_sora_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******open_sora train end***********"


echo "*******open_sora text to video begin***********"
(bash text_to_video.sh) 2>&1 | tee ${log_dir}/open_sora_text_to_video.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "open_sora_text_to_video run success" >>"${log_dir}/ce_res.log"
else
    echo "open_sora text_to_video run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******open_sora text_to_video end***********"


echo "*******open_sora image condition begin***********"
(bash image_condiction_video.sh) 2>&1 | tee ${log_dir}/open_sora_image_condiction_video.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "open_sora_image_condiction_video run success" >>"${log_dir}/ce_res.log"
else
    echo "open_sora image_condiction_video run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******open_sora image_condiction_video end***********"


echo "*******open_sora video_connection begin***********"
(bash video_connection.sh) 2>&1 | tee ${log_dir}/open_sora_video_connection.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "open_sora_video_connection run success" >>"${log_dir}/ce_res.log"
else
    echo "open_sora video_connection run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******open_sora video_connection end***********"



echo "*******open_sora video_extend_edit begin***********"
(bash video_extend_edit.sh) 2>&1 | tee ${log_dir}/open_sora_video_extend_edit.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "open_sora_video_extend_edit run success" >>"${log_dir}/ce_res.log"
else
    echo "open_sora video_extend_edit run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******open_sora video_extend_edit end***********"


# # 查看结果
# cat ${log_dir}/ce_res.log
rm -rf ${work_path}/OpenSoraData/*

echo exit_code:${exit_code}
exit ${exit_code}
