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

echo "*******ppdiffusers/examples/inference latent_diffusion text_to_image_generation begin***********"
(python text_to_image_generation-latent_diffusion.py) 2>&1 | tee ${log_dir}/inference_latent_diffusion_text_to_image_generation.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/examples/inference latent_diffusion text_to_image_generation run success" >>"${log_dir}/res.log"
else
    echo "ppdiffusers/examples/inference latent_diffusion text_to_image_generation run fail" >>"${log_dir}/res.log"
fi
echo "*******ppdiffusers/examples/inference latent_diffusion text_to_image_generation end***********"

echo "*******ppdiffusers/examples/inference latent_diffusion super_resolution begin***********"
(python super_resolution-latent_diffusion.py) 2>&1 | tee ${log_dir}/inference_latent_diffusion_super_resolution.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/examples/inference latent_diffusion super_resolution run success" >>"${log_dir}/res.log"
else
    echo "ppdiffusers/examples/inference latent_diffusion super_resolution run fail" >>"${log_dir}/res.log"
fi
echo "*******ppdiffusers/examples/inference latent_diffusion super_resolution end***********"

echo "*******ppdiffusers/examples/inference latent_diffusion unconditional_image_generation begin***********"
(python unconditional_image_generation-latent_diffusion_uncond.py) 2>&1 | tee ${log_dir}/inference_latent_diffusion_unconditional_image_generation.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/examples/inference latent_diffusion unconditional_image_generation run success" >>"${log_dir}/res.log"
else
    echo "ppdiffusers/examples/inference latent_diffusion unconditional_image_generation run fail" >>"${log_dir}/res.log"
fi
echo "*******ppdiffusers/examples/inference latent_diffusion unconditional_image_generation end***********"

echo exit_code:${exit_code}
exit ${exit_code}
