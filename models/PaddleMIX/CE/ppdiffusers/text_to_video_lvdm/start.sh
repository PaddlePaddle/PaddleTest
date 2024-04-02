#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/examples/text_to_video_lvdm
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
echo "*******text_to_video_lvdm unconditional_generationsinge_train begin***********"
(bash unconditional_generation_single_train.sh) 2>&1 | tee ${log_dir}/text_to_video_lvdm_unconditional_generation_singe_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "text_to_video_lvdm unconditional_generation_singe_train run success" >>"${log_dir}/ce_res.log"
else
    echo "text_to_video_lvdm unconditional_generation_singe_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******text_to_video_lvdm unconditional_generation_singe_train end***********"

# echo "*******text_to_video_lvdm text2video_generation_single_train begin***********"
# (bash text2video_generation_single_train.sh) 2>&1 | tee ${log_dir}/text_to_video_lvdm_text2video_generation_single_train.log
# tmp_exit_code=${PIPESTATUS[0]}
# exit_code=$(($exit_code + ${tmp_exit_code}))
# if [ ${tmp_exit_code} -eq 0 ]; then
#     echo "text_to_video_lvdm text2video_generation_single_train run success" >>"${log_dir}/ce_res.log"
# else
#     echo "text_to_video_lvdm text2video_generation_single_train run fail" >>"${log_dir}/ce_res.log"
# fi
# echo "*******text_to_video_lvdm text2video_generation_single_train end***********"


# 多机训练
echo "*******text_to_video_lvdm unconditional_generation_muti_train begin***********"
(bash unconditional_generation_multi_train.sh) 2>&1 | tee ${log_dir}/text_to_video_lvdm_unconditional_generation_multi_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "text_to_video_lvdm unconditional_generation_muti_train run success" >>"${log_dir}/ce_res.log"
else
    echo "text_to_video_lvdm unconditional_generation_muti_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******text_to_video_lvdm unconditional_generation_muti_train end***********"

# echo "*******text_to_video_lvdm text2video_generation_multi_train begin***********"
# (bash text2video_generation_multi_train.sh) 2>&1 | tee ${log_dir}/text_to_video_lvdm_text2video_generation_multi_train.log
# tmp_exit_code=${PIPESTATUS[0]}
# exit_code=$(($exit_code + ${tmp_exit_code}))
# if [ ${tmp_exit_code} -eq 0 ]; then
#     echo "text_to_video_lvdm text2video_generation_multi_train run success" >>"${log_dir}/ce_res.log"
# else
#     echo "text_to_video_lvdm text2video_generation_multi_train run fail" >>"${log_dir}/ce_res.log"
# fi
# echo "*******text_to_video_lvdm text2video_generation_multi_train end***********"

# # 查看结果
# cat ${log_dir}/ce_res.log
rm -rf ${work_path}/sky_timelapse_lvdm/*
rm -rf ${work_path}/temp/checkpoints_text2video/*
rm -rf ${work_path}/temp/checkpoints_short/*

echo exit_code:${exit_code}
exit ${exit_code}
