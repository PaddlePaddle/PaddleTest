#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

log_dir=${root_path}/log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

exit_code=0

export USE_PPXFORMERS=False
(python export_model.py \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --output_path static_model/stable-diffusion-v1-5) 2>&1 | tee ${log_dir}/sd_deploy_export_model.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy sd_deploy_export_model success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy sd_deploy_export_model fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy sd_deploy_export_model end***********"

rm -rf infer_op_raw_fp16
rm -rf infer_op_zero_copy_infer_fp16

(python infer.py \
    --model_dir static_model/stable-diffusion-v1-5 \
    --scheduler "preconfig-euler-ancestral" \
    --backend paddle \
    --device gpu \
    --task_name text2img) 2>&1 | tee ${log_dir}/paddle_sd_deploy_infer_text2img.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy paddle sd_deploy_infer_text2img success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy paddle sd_deploy_infer_text2img fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy paddle sd_deploy_infer_text2img end***********"

(python infer.py \
    --model_dir static_model/stable-diffusion-v1-5 \
    --scheduler "preconfig-euler-ancestral" \
    --backend paddle \
    --device gpu \
    --task_name img2img) 2>&1 | tee ${log_dir}/paddle_sd_deploy_infer_img2img.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy paddle sd_deploy_infer_img2img success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy paddle sd_deploy_infer_img2img fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy paddle sd_deploy_infer_img2img end***********"

(python infer.py \
    --model_dir static_model/stable-diffusion-v1-5 \
    --scheduler "preconfig-euler-ancestral" \
    --backend paddle \
    --device gpu \
    --task_name inpaint_legacy) 2>&1 | tee ${log_dir}/paddle_sd_deploy_infer_inpaint_legacy.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy paddle sd_deploy_infer_inpaint_legacy success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy paddle sd_deploy_infer_inpaint_legacy fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy paddle sd_deploy_infer_inpaint_legacy end***********"

# paddle_tensorrt
(python infer.py \
    --model_dir static_model/stable-diffusion-v1-5 \
    --scheduler "preconfig-euler-ancestral" \
    --backend paddle_tensorrt \
    --device gpu \
    --task_name text2img) 2>&1 | tee ${log_dir}/paddle_tensorrt_sd_infer_text2img.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy paddle_tensorrt sd_deploy_infer_text2img success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy paddle_tensorrt sd_deploy_infer_text2img fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy paddle_tensorrt sd_deploy_infer_text2img end***********"

(python infer.py \
    --model_dir static_model/stable-diffusion-v1-5 \
    --scheduler "preconfig-euler-ancestral" \
    --backend paddle_tensorrt \
    --device gpu \
    --task_name img2img) 2>&1 | tee ${log_dir}/paddle_tensorrt_sd_deploy_infer_img2img.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy paddle_tensorrt sd_deploy_infer_img2img success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy paddle_tensorrt sd_deploy_infer_img2img fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy paddle_tensorrt sd_deploy_infer_img2img end***********"

(python infer.py \
    --model_dir static_model/stable-diffusion-v1-5 \
    --scheduler "preconfig-euler-ancestral" \
    --backend paddle_tensorrt \
    --device gpu \
    --task_name inpaint_legacy) 2>&1 | tee ${log_dir}/paddle_tensorrt_sd_deploy_infer_inpaint_legacy.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy paddle_tensorrt sd_deploy_infer_inpaint_legacy success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy paddle_tensorrt sd_deploy_infer_inpaint_legacy fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy paddle_tensorrt sd_deploy_infer_inpaint_legacy end***********"

(python ./utils/test_image_diff.py \
    --source_image ./infer_op_raw_fp16/text2img.png \
    --target_image https://paddlenlp.bj.bcebos.com/models/community/baicai/sd15_infer_op_raw_fp16/text2img.png) 2>&1 | tee ${log_dir}/sd_deploy_test_image_diff_text2img.log
python ${cur_path}/annalyse_log_tool.py --file_path ${log_dir}/sd_deploy_test_image_diff_text2img.log
tmp_exit_code=$?
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy sd_deploy_test_image_diff_text2img success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy sd_deploy_test_image_diff_text2img fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy sd_deploy_test_image_diff_text2img end***********"

(python ./utils/test_image_diff.py \
    --source_image ./infer_op_raw_fp16/img2img.png \
    --target_image https://paddlenlp.bj.bcebos.com/models/community/baicai/sd15_infer_op_raw_fp16/img2img.png) 2>&1 | tee ${log_dir}/sd_deploy_test_image_diff_img2img.log
python ${cur_path}/annalyse_log_tool.py --file_path ${log_dir}/sd_deploy_test_image_diff_img2img.log
tmp_exit_code=$?
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy sd_deploy_test_image_diff_img2img success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy sd_deploy_test_image_diff_img2img fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy sd_deploy_test_image_diff_img2img end***********"

(python ./utils/test_image_diff.py \
    --source_image ./infer_op_raw_fp16/inpaint_legacy.png \
    --target_image https://paddlenlp.bj.bcebos.com/models/community/baicai/sd15_infer_op_raw_fp16/inpaint_legacy.png) 2>&1 | tee ${log_dir}/sd_deploy_test_image_diff_inpaint.log
python ${cur_path}/annalyse_log_tool.py --file_path ${log_dir}/sd_deploy_test_image_diff_inpaint.log
tmp_exit_code=$?
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy sd_deploy_test_image_diff_inpaint success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy sd_deploy_test_image_diff_inpaint fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy sd_deploy_test_image_diff_inpaint end***********"

echo exit_code:${exit_code}
exit ${exit_code}
