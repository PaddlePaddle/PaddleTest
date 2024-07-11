#!/bin/bash

log_dir=${root_path}/deploy_log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

export FLAGS_use_cuda_managed_memory=true
export USE_PPXFORMERS=False

# text2img
(python infer.py \
    --model_dir static_model/stable-diffusion-v1-5-ipadapter/ \
    --scheduler "ddim" \
    --backend paddle \
    --device gpu \
    --task_name text2img) 2>&1 | tee ${log_dir}/ipadapter_sd15_inference_text2img.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/sd15 ipadapter_sd15_inference_text2img success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/sd15 ipadapter_sd15_inference_text2img fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/sd15 ipadapter_sd15_inference_text2img end***********"

# img2img
(python infer.py \
    --model_dir static_model/stable-diffusion-v1-5-ipadapter/ \
    --scheduler "ddim" \
    --backend paddle \
    --device gpu \
    --task_name img2img) 2>&1 | tee ${log_dir}/ipadapter_sd15_inference_img2img.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/sd15 ipadapter_sd15_inference_img2img success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/sd15 ipadapter_sd15_inference_img2img fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/sd15 ipadapter_sd15_inference_img2img end***********"

# inpaint
(python infer.py \
    --model_dir static_model/stable-diffusion-v1-5-ipadapter/ \
    --scheduler "ddim" \
    --backend paddle \
    --device gpu \
    --task_name inpaint) 2>&1 | tee ${log_dir}/ipadapter_sd15_inference_inpaint.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/sd15 ipadapter_sd15_inference_inpaint success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/sd15 ipadapter_sd15_inference_inpaint fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/sd15 ipadapter_sd15_inference_inpaint end***********"

# tensorrt
# text2img
(python infer.py \
    --model_dir static_model/stable-diffusion-v1-5-ipadapter/ \
    --scheduler "ddim" \
    --backend paddle_tensorrt \
    --device gpu \
    --task_name text2img) 2>&1 | tee ${log_dir}/ipadapter_sd15_inference_tensorrt_text2img.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/sd15 ipadapter_sd15_inference_tensorrt_text2img success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/sd15 ipadapter_sd15_inference_tensorrt_text2img fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/sd15 ipadapter_sd15_inference_tensorrt_text2img end***********"

# img2img
(python infer.py \
    --model_dir static_model/stable-diffusion-v1-5-ipadapter/ \
    --scheduler "ddim" \
    --backend paddle_tensorrt \
    --device gpu \
    --task_name img2img) 2>&1 | tee ${log_dir}/ipadapter_sd15_inference_tensorrt_img2img.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/sd15 ipadapter_sd15_inference_tensorrt_img2img success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/sd15 ipadapter_sd15_inference_tensorrt_img2img fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/sd15 ipadapter_sd15_inference_tensorrt_img2img end***********"

# inpaint
(python infer.py \
    --model_dir static_model/stable-diffusion-v1-5-ipadapter/ \
    --scheduler "ddim" \
    --backend paddle_tensorrt \
    --device gpu \
    --task_name inpaint) 2>&1 | tee ${log_dir}/ipadapter_sd15_inference_tensorrt_inpaint.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/sd15 ipadapter_sd15_inference_tensorrt_inpaint success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/sd15 ipadapter_sd15_inference_tensorrt_inpaint fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/sd15 ipadapter_sd15_inference_tensorrt_inpaint end***********"

echo exit_code:${exit_code}
exit ${exit_code}
