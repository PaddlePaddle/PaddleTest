#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/examples/inference/
echo ${work_path}

log_dir=${root_path}/log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

exit_code=0

cd ${work_path}

echo "*******ppdiffusers/examples/inference semantic_stable_diffusion text_guided_generation begin***********"
(python text_guided_generation-semantic_stable_diffusion.py) 2>&1 | tee ${log_dir}/inference_semantic_stable_diffusion_text_guided_generation.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/examples/inference semantic_stable_diffusion text_guided_generation run success" >>"${log_dir}/res.log"
else
    echo "ppdiffusers/examples/inference semantic_stable_diffusion text_guided_generation run fail" >>"${log_dir}/res.log"
fi
echo "*******ppdiffusers/examples/inference semantic_stable_diffusion text_guided_generation end***********"

echo "*******ppdiffusers/examples/inference stable_diffusion text_to_image_generation begin***********"
(python text_to_image_generation-stable_diffusion.py) 2>&1 | tee ${log_dir}/inference_stable_diffusion_text_to_image_generation.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/examples/inference stable_diffusion text_to_image_generation run success" >>"${log_dir}/res.log"
else
    echo "ppdiffusers/examples/inference stable_diffusion text_to_image_generation run fail" >>"${log_dir}/res.log"
fi
echo "*******ppdiffusers/examples/inference stable_diffusion text_to_image_generation end***********"

echo "*******ppdiffusers/examples/inference stable_diffusion image_to_image_text_guided_generation begin***********"
(python image_to_image_text_guided_generation-stable_diffusion.py) 2>&1 | tee ${log_dir}/inference_stable_diffusion_image_to_image_text_guided_generation.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/examples/inference stable_diffusion image_to_image_text_guided_generation run success" >>"${log_dir}/res.log"
else
    echo "ppdiffusers/examples/inference stable_diffusion image_to_image_text_guided_generation run fail" >>"${log_dir}/res.log"
fi
echo "*******ppdiffusers/examples/inference stable_diffusion image_to_image_text_guided_generation end***********"

echo "*******ppdiffusers/examples/inference stable_diffusion text_guided_image_inpainting begin***********"
(python text_guided_image_inpainting-stable_diffusion.py) 2>&1 | tee ${log_dir}/inference_stable_diffusion_text_guided_image_inpainting.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/examples/inference stable_diffusion text_guided_image_inpainting run success" >>"${log_dir}/res.log"
else
    echo "ppdiffusers/examples/inference stable_diffusion text_guided_image_inpainting run fail" >>"${log_dir}/res.log"
fi
echo "*******ppdiffusers/examples/inference stable_diffusion text_guided_image_inpainting end***********"

echo "*******ppdiffusers/examples/inference stable_diffusion  text_to_image_generation begin***********"
(python text_to_image_generation-stable_diffusion_2.py) 2>&1 | tee ${log_dir}/inference_stable_diffusion_text_to_image_generation.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/examples/inference stable_diffusion text_to_image_generation run success" >>"${log_dir}/res.log"
else
    echo "ppdiffusers/examples/inference stable_diffusion text_to_image_generation run fail" >>"${log_dir}/res.log"
fi
echo "*******ppdiffusers/examples/inference stable_diffusion text_to_image_generation end***********"

echo "*******ppdiffusers/examples/inference stable_diffusion image_to_image_text_guided_generation begin***********"
(python image_to_image_text_guided_generation-stable_diffusion_2.py) 2>&1 | tee ${log_dir}/inference_stable_diffusion_image_to_image_text_guided_generation.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/examples/inference stable_diffusion image_to_image_text_guided_generation run success" >>"${log_dir}/res.log"
else
    echo "ppdiffusers/examples/inference stable_diffusion image_to_image_text_guided_generation run fail" >>"${log_dir}/res.log"
fi
echo "*******ppdiffusers/examples/inference stable_diffusion image_to_image_text_guided_generation end***********"

echo "*******ppdiffusers/examples/inference stable_diffusion2 text_guided_image_inpainting begin***********"
(python text_guided_image_inpainting-stable_diffusion_2.py) 2>&1 | tee ${log_dir}/inference_stable_diffusion2_text_guided_image_inpainting.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/examples/inference stable_diffusion2 text_guided_image_inpainting run success" >>"${log_dir}/res.log"
else
    echo "ppdiffusers/examples/inference stable_diffusion2 text_guided_image_inpainting run fail" >>"${log_dir}/res.log"
fi
echo "*******ppdiffusers/examples/inference stable_diffusion2 text_guided_image_inpainting end***********"

echo "*******ppdiffusers/examples/inference stable_diffusion2 text_guided_image_upscaling begin***********"
(python text_guided_image_upscaling-stable_diffusion_2.py) 2>&1 | tee ${log_dir}/inference_stable_diffusion2_text_guided_image_upscaling.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/examples/inference stable_diffusion2 text_guided_image_upscaling run success" >>"${log_dir}/res.log"
else
    echo "ppdiffusers/examples/inference stable_diffusion2 text_guided_image_upscaling run fail" >>"${log_dir}/res.log"
fi
echo "*******ppdiffusers/examples/inference stable_diffusion2 text_guided_image_upscaling end***********"

echo "*******ppdiffusers/examples/inference stable_diffusion_safe text_to_image_generation begin***********"
(python text_to_image_generation-stable_diffusion_safe.py) 2>&1 | tee ${log_dir}/inference_stable_diffusion_safe_text_to_image_generation.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/examples/inference stable_diffusion_safe text_to_image_generation run success" >>"${log_dir}/res.log"
else
    echo "ppdiffusers/examples/inference stable_diffusion_safe text_to_image_generation run fail" >>"${log_dir}/res.log"
fi
echo "*******ppdiffusers/examples/inference stable_diffusion_safe text_to_image_generation end***********"

echo exit_code:${exit_code}
exit ${exit_code}
