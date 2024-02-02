#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

log_dir=${root_path}/log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

exit_code=0

export USE_PPXFORMERS=False

# export model
(python export_model.py \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
    --output_path static_model/stable-diffusion-xl-base-1.0) 2>&1 | tee ${log_dir}/sdxl_export_model.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/sdxl sdxl_export_model success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/sdxl sdxl_export_model fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/sdxl sdxl_export_model end***********"

rm -rf infer_op_raw_fp16

# inference
export FLAGS_use_cuda_managed_memory=true
(python infer.py \
    --model_dir static_model/stable-diffusion-xl-base-1.0 \
    --scheduler "preconfig-euler-ancestral" \
    --backend paddle \
    --device gpu \
    --task_name all) 2>&1 | tee ${log_dir}/paddle_sdxl_inference.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/sdxl paddle sdxl_inference success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/sdxl paddle sdxl_inference fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/sdxl paddle sdxl_inference end***********"

(python ../utils/test_image_diff.py \
    --source_image ./infer_op_raw_fp16/text2img.png \
    --target_image https://paddlenlp.bj.bcebos.com/models/community/baicai/sdxl_infer_op_raw_fp16/text2img.png) 2>&1 | tee ${log_dir}/sdxl_test_image_diff_text2img.log
python ${cur_path}/annalyse_log_tool.py \
    --file_path ${log_dir}/sdxl_test_image_diff_text2img.log
tmp_exit_code=$?
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/sdxl sdxl_test_image_diff_text2img success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/sdxl sdxl_test_image_diff_text2img fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/sdxl sdxl_test_image_diff_text2img end***********"

(python ../utils/test_image_diff.py \
    --source_image ./infer_op_raw_fp16/img2img.png \
    --target_image https://paddlenlp.bj.bcebos.com/models/community/baicai/sdxl_infer_op_raw_fp16/img2img.png) 2>&1 | tee ${log_dir}/sdxl_test_image_diff_img2img.log
python ${cur_path}/annalyse_log_tool.py --file_path ${log_dir}/sdxl_test_image_diff_img2img.log
tmp_exit_code=$?
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/sdxl sdxl_test_image_diff_img2img success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/sdxl sdxl_test_image_diff_img2img fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/sdxl sdxl_test_image_diff_img2img end***********"

(python ../utils/test_image_diff.py \
    --source_image ./infer_op_raw_fp16/inpaint.png \
    --target_image https://paddlenlp.bj.bcebos.com/models/community/baicai/sdxl_infer_op_raw_fp16/inpaint.png) 2>&1 | tee ${log_dir}/sdxl_test_image_diff_inpaint.log
python ${cur_path}/annalyse_log_tool.py --file_path ${log_dir}/sdxl_test_image_diff_inpaint.log
tmp_exit_code=$?
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/sdxl sdxl_test_image_diff_inpaint success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/sdxl sdxl_test_image_diff_inpaint fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/sdxl sdxl_test_image_diff_inpaint end***********"

# paddle_tensorrt
(python infer.py \
    --model_dir static_model/stable-diffusion-xl-base-1.0 \
    --scheduler "preconfig-euler-ancestral" \
    --backend paddle_tensorrt \
    --device gpu \
    --task_name all \
    --infer_op raw) 2>&1 | tee ${log_dir}/sdxl_inference_paddle_tensorrt.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/sdxl sdxl_inference_paddle_tensorrt success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/sdxl sdxl_inference_paddle_tensorrt fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/sdxl sdxl_inference_paddle_tensorrt end***********"

echo exit_code:${exit_code}
exit ${exit_code}
