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
    --controlnet_pretrained_model_name_or_path lllyasviel/sd-controlnet-canny \
    --output_path static_model/stable-diffusion-v1-5-canny) 2>&1 | tee ${log_dir}/sd_controlnet_export_model.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/controlnet sd_controlnet_export_model success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/controlnet sd_controlnet_export_model fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/controlnet sd_controlnet_export_model end***********"

rm -rf infer_op_raw_fp16
rm -rf infer_op_zero_copy_infer_fp16

# paddle
(python infer.py \
    --model_dir static_model/stable-diffusion-v1-5-canny \
    --scheduler "preconfig-euler-ancestral" \
    --backend paddle --device gpu \
    --task_name text2img_control) 2>&1 | tee ${log_dir}/paddle_sd_controlnet_infer_text2img_control.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/controlnet paddle sd_controlnet_infer_text2img_control success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/controlnet paddle sd_controlnet_infer_text2img_control fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/controlnet paddle sd_controlnet_infer_text2img_control end***********"

(python infer.py \
    --model_dir static_model/stable-diffusion-v1-5-canny \
    --scheduler "preconfig-euler-ancestral" \
    --backend paddle \
    --device gpu \
    --task_name img2img_control) 2>&1 | tee ${log_dir}/sd_controlnet_infer_img2img_control.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/controlnet sd_controlnet_infer_img2img_control success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/controlnet sd_controlnet_infer_img2img_control fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/controlnet sd_controlnet_infer_img2img_control end***********"

(python infer.py \
    --model_dir static_model/stable-diffusion-v1-5-canny \
    --scheduler "preconfig-euler-ancestral" \
    --backend paddle \
    --device gpu \
    --task_name inpaint_legacy_control) 2>&1 | tee ${log_dir}/sd_controlnet_infer_inpaint_legacy_control.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/controlnet sd_controlnet_infer_inpaint_legacy_control success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/controlnet sd_controlnet_infer_inpaint_legacy_control fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/controlnet sd_controlnet_infer_inpaint_legacy_control end***********"

# paddle_tensorrt
(python infer.py \
    --model_dir static_model/stable-diffusion-v1-5-canny \
    --scheduler "preconfig-euler-ancestral" \
    --backend paddle_tensorrt \
    --device gpu \
    --task_name text2img_control) 2>&1 | tee ${log_dir}/paddle_tensorrt_sd_controlnet_infer_inpaint_legacy_control.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/controlnet paddle_tensorrt sd_controlnet_infer_inpaint_legacy_control success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/controlnet paddle_tensorrt sd_controlnet_infer_inpaint_legacy_control fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/controlnet paddle_tensorrt sd_infer_inpaint_legacy_control end***********"

(python infer.py \
    --model_dir static_model/stable-diffusion-v1-5-canny \
    --scheduler "preconfig-euler-ancestral" \
    --backend paddle_tensorrt \
    --device gpu \
    --task_name img2img_control) 2>&1 | tee ${log_dir}/paddle_tensorrt_sd_controlnet_infer_img2img_control.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/controlnet paddle_tensorrt paddle_tensorrt_sd_controlnet_infer_img2img_control success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/controlnet paddle_tensorrt paddle_tensorrt_sd_controlnet_infer_img2img_control fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/controlnet paddle_tensorrt paddle_tensorrt_sd_controlnet_infer_img2img_control end***********"

(python infer.py \
    --model_dir static_model/stable-diffusion-v1-5-canny \
    --scheduler "preconfig-euler-ancestral" \
    --backend paddle_tensorrt \
    --device gpu \
    --task_name inpaint_legacy_control) 2>&1 | tee ${log_dir}/paddle_tensorrt_sd_controlnet_inpaint_legacy_control.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/controlnet paddle_tensorrt paddle_tensorrt_sd_controlnet_inpaint_legacy_control success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/controlnet paddle_tensorrt paddle_tensorrt_sd_controlnet_inpaint_legacy_control fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/controlnet paddle_tensorrt paddle_tensorrt_sd_controlnet_inpaint_legacy_control end***********"

(python ../utils/test_image_diff.py \
    --source_image ./infer_op_raw_fp16/text2img_control.png \
    --target_image https://paddlenlp.bj.bcebos.com/models/community/baicai/sd15_controlnet_infer_op_raw_fp16/text2img_control.png) 2>&1 | tee ${log_dir}/sd_controlnet_test_image_diff_text2img.log
python ${cur_path}/annalyse_log_tool.py \
    --file_path ${log_dir}/sd_controlnet_test_image_diff_text2img.log
tmp_exit_code=$?
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/controlnet sd_controlnet_test_image_diff_text2img success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/controlnet sd_controlnet_test_image_diff_text2img fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/controlnet sd_controlnet_test_image_diff_text2img end***********"

(python ../utils/test_image_diff.py \
    --source_image ./infer_op_raw_fp16/img2img_control.png \
    --target_image https://paddlenlp.bj.bcebos.com/models/community/baicai/sd15_controlnet_infer_op_raw_fp16/img2img_control.png) 2>&1 | tee ${log_dir}/sd_controlnet_test_image_diff_img2img.log
python ${cur_path}/annalyse_log_tool.py --file_path ${log_dir}/sd_controlnet_test_image_diff_img2img.log
tmp_exit_code=$?
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/controlnet sd_controlnet_test_image_diff_img2img success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/controlnet sd_controlnet_test_image_diff_img2img fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/controlnet sd_controlnet_test_image_diff_img2img end***********"

(python ../utils/test_image_diff.py \
    --source_image ./infer_op_raw_fp16/inpaint_legacy_control.png \
    --target_image https://paddlenlp.bj.bcebos.com/models/community/baicai/sd15_controlnet_infer_op_raw_fp16/inpaint_legacy_control.png) 2>&1 | tee ${log_dir}/sd_controlnet_test_image_diff_inpaint.log
python ${cur_path}/annalyse_log_tool.py --file_path ${log_dir}/sd_controlnet_test_image_diff_inpaint.log
tmp_exit_code=$?
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/controlnet sd_controlnet_test_image_diff_inpaint success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/controlnet sd_controlnet_test_image_diff_inpaint fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/controlnet sd_controlnet_test_image_diff_inpaint end***********"

echo exit_code:${exit_code}
exit ${exit_code}
