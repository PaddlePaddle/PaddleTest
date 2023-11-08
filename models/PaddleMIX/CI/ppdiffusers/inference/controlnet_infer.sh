#!/bin/bash

cur_path=`pwd`
echo ${cur_path}


work_path=${root_path}/PaddleMIX/ppdiffusers/examples/inference/
echo ${work_path}

log_dir=${root_path}/log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

exit_code=0

cd ${work_path}


echo "*******ppdiffusers/examples/inference controlnet image_to_image_text_guided_generation begin***********"
(python image_to_image_text_guided_generation-controlnet.py) 2>&1 | tee ${log_dir}/inference_controlnet_image_to_image_text_guided_generation.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers/examples/inference controlnet image_to_image_text_guided_generation run success" >> "${log_dir}/res.log"
else
    echo "ppdiffusers/examples/inference controlnet image_to_image_text_guided_generation run fail" >> "${log_dir}/res.log"
fi
echo "*******ppdiffusers/examples/inference controlnet image_to_image_text_guided_generation end***********"


echo exit_code:${exit_code}
exit ${exit_code}