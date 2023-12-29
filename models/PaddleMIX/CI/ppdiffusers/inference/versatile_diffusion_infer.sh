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

# echo "*******ppdiffusers/examples/inference versatile_diffusion text_to_image_generation begin***********"
# (python text_to_image_generation-versatile_diffusion.py) 2>&1 | tee ${log_dir}/inference_versatile_diffusion_text_to_image_generation.log
# tmp_exit_code=${PIPESTATUS[0]}
# exit_code=$(($exit_code + ${tmp_exit_code}))
# if [ ${tmp_exit_code} -eq 0 ]; then
#     echo "ppdiffusers/examples/inference versatile_diffusion text_to_image_generation run success" >> "${log_dir}/res.log"
# else
#     echo "ppdiffusers/examples/inference versatile_diffusion text_to_image_generation run fail" >> "${log_dir}/res.log"
# fi
# echo "*******ppdiffusers/examples/inference versatile_diffusion text_to_image_generation end***********"

echo "*******ppdiffusers/examples/inference versatile_diffusion image_variation begin***********"
(python image_variation-versatile_diffusion.py) 2>&1 | tee ${log_dir}/inference_versatile_diffusion_image_variation.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/examples/inference versatile_diffusion image_variation run success" >>"${log_dir}/res.log"
else
    echo "ppdiffusers/examples/inference versatile_diffusion image_variation run fail" >>"${log_dir}/res.log"
fi
echo "*******ppdiffusers/examples/inference versatile_diffusion image_variation end***********"

# echo "*******ppdiffusers/examples/inference versatile_diffusion dual_text_and_image_guided_generation begin***********"
# (python dual_text_and_image_guided_generation-versatile_diffusion.py) 2>&1 | tee ${log_dir}/inference_versatile_diffusion_dual_text_and_image_guided_generation.log
# tmp_exit_code=${PIPESTATUS[0]}
# exit_code=$(($exit_code + ${tmp_exit_code}))
# if [ ${tmp_exit_code} -eq 0 ]; then
#     echo "ppdiffusers/examples/inference versatile_diffusion dual_text_and_image_guided_generation run success" >> "${log_dir}/res.log"
# else
#     echo "ppdiffusers/examples/inference versatile_diffusion dual_text_and_image_guided_generation run fail" >> "${log_dir}/res.log"
# fi
# echo "*******ppdiffusers/examples/inference versatile_diffusion dual_text_and_image_guided_generation end***********"

echo exit_code:${exit_code}
exit ${exit_code}
