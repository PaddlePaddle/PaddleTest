#!/bin/bash

log_dir=${root_path}/deploy_log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

exit_code=0

(python infer.py \
    --model_dir static_model/stable-diffusion-v1-5-ipadapter/ \
    --scheduler "ddim" \
    --backend paddle \
    --device gpu \
    --task_name text2img) 2>&1 | tee ${log_dir}/ipadapter_sd15_text2img_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/ipadapter/sd15 ipadapter_sd15_text2img_infer success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/ipadapter/sd15 ipadapter_sd15_text2img_infer fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/ipadapter/sd15 ipadapter_sd15_text2img_infer end***********"

(python infer.py \
    --model_dir static_model/stable-diffusion-v1-5-ipadapter/ \
    --scheduler "ddim" \
    --backend paddle \
    --device gpu \
    --task_name img2img) 2>&1 | tee ${log_dir}/ipadapter_sd15_img2img_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/ipadapter/sd15 ipadapter_sd15_img2img_infer success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/ipadapter/sd15 ipadapter_sd15_img2img_infer fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/ipadapter/sd15 ipadapter_sd15_img2img_infer end***********"

(python infer.py \
    --model_dir static_model/stable-diffusion-v1-5-ipadapter/ \
    --scheduler "ddim" \
    --backend paddle \
    --device gpu \
    --task_name inpaint) 2>&1 | tee ${log_dir}/ipadapter_sd15_inpaint_infer.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/deploy/ipadapter/sd15 ipadapter_sd15_inpaint_infer success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers/deploy/ipadapter/sd15 ipadapter_sd15_inpaint_infer fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers/deploy/ipadapter/sd15 ipadapter_sd15_inpaint_infer end***********"

echo exit_code:${exit_code}
exit ${exit_code}
