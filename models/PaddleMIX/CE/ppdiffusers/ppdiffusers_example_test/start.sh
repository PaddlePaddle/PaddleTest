#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}


log_dir=${root_path}/examples_log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

exit_code=0

work_path=${root_path}/PaddleMIX/ppdiffusers/examples/dreambooth/
echo ${work_path}
/bin/cp -rf dreambooth_test.sh ${work_path}

cd ${work_path}

echo "*******examples_test dreambooth_test begin***********"
(bash dreambooth_test.sh) 2>&1 | tee ${log_dir}/dreambooth_test.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "examples_test dreambooth_test run success" >>"${log_dir}/ce_res.log"
else
    echo "examples_test dreambooth_test run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******examples_test dreambooth_test end***********"

cd ${cur_path}
echo ${cur_path}
work_path=${root_path}/PaddleMIX/ppdiffusers/examples/text_to_image/
echo ${work_path}
/bin/cp -rf text_to_image_test.sh ${work_path}

cd ${work_path}
echo "*******examples_test text_to_image_test begin***********"
(bash text_to_image_test.sh) 2>&1 | tee ${log_dir}/text_to_image_test.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "examples_test text_to_image_test run success" >>"${log_dir}/ce_res.log"
else
    echo "examples_test text_to_image_test run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******examples_test text_to_image_test end***********"

cd ${cur_path}
echo ${cur_path}
work_path=${root_path}/PaddleMIX/ppdiffusers/examples/textual_inversion/
echo ${work_path}
/bin/cp -rf textual_inversion_test.sh ${work_path}

cd ${work_path}
echo "*******examples_test textual_inversion_test begin***********"
(bash textual_inversion_test.sh) 2>&1 | tee ${log_dir}/textual_inversion_test.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "examples_test textual_inversion_test run success" >>"${log_dir}/ce_res.log"
else
    echo "examples_test textual_inversion_test run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******examples_test textual_inversion_test end***********"

cd ${cur_path}
echo ${cur_path}
work_path=${root_path}/PaddleMIX/ppdiffusers/examples/text_to_image_laion400m/
echo ${work_path}
/bin/cp -rf text_to_image_laion400m_test.sh ${work_path}

cd ${work_path}
echo "*******examples_test textual_inversion_test begin***********"
(bash text_to_image_laion400m_test.sh) 2>&1 | tee ${log_dir}/text_to_image_laion400m_test.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "examples_test text_to_image_laion400m_test run success" >>"${log_dir}/ce_res.log"
else
    echo "examples_test text_to_image_laion400m_test run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******examples_test text_to_image_laion400m_test end***********"

cd ${cur_path}
echo ${cur_path}
work_path=${root_path}/PaddleMIX/ppdiffusers/examples/stable_diffusion/
echo ${work_path}
/bin/cp -rf stable_diffusion_test.sh ${work_path}

cd ${work_path}
echo "*******examples_test stable_diffusion_test begin***********"
(bash stable_diffusion_test.sh) 2>&1 | tee ${log_dir}/stable_diffusion_test.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "examples_test stable_diffusion_test run success" >>"${log_dir}/ce_res.log"
else
    echo "examples_test stable_diffusion_test run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******examples_test stable_diffusion_test end***********"

cd ${cur_path}
echo ${cur_path}
work_path=${root_path}/PaddleMIX/ppdiffusers/examples/autoencoder/vae
echo ${work_path}
/bin/cp -rf autoencoder_vae_test.sh ${work_path}

cd ${work_path}
echo "*******examples_test autoencoder_vae_test begin***********"
(bash autoencoder_vae_test.sh) 2>&1 | tee ${log_dir}/autoencoder_vae_test.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "examples_test autoencoder_vae_test run success" >>"${log_dir}/ce_res.log"
else
    echo "examples_test autoencoder_vae_test run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******examples_test autoencoder_vae_test end***********"

cd ${cur_path}
echo ${cur_path}
work_path=${root_path}/PaddleMIX/ppdiffusers/examples/controlnet
echo ${work_path}
/bin/cp -rf controlnet_test.sh ${work_path}

cd ${work_path}
echo "*******examples_test controlnet_test begin***********"
(bash controlnet_test.sh) 2>&1 | tee ${log_dir}/controlnet_test.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "examples_test controlnet_test run success" >>"${log_dir}/ce_res.log"
else
    echo "examples_test controlnet_test run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******examples_test controlnet_test end***********"

cd ${cur_path}
echo ${cur_path}
work_path=${root_path}/PaddleMIX/ppdiffusers/examples/t2i-adapter
echo ${work_path}
/bin/cp -rf t2i_adapter_test.sh ${work_path}

cd ${work_path}
echo "*******examples_test t2i_adapter_test begin***********"
(bash t2i_adapter_test.sh) 2>&1 | tee ${log_dir}/t2i_adapter_test.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "examples_test t2i_adapter_test run success" >>"${log_dir}/ce_res.log"
else
    echo "examples_test t2i_adapter_test run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******examples_test t2i_adapter_test end***********"

cd ${cur_path}
echo ${cur_path}
work_path=${root_path}/PaddleMIX/ppdiffusers/examples/ip_adapter
echo ${work_path}
/bin/cp -rf ip_adapter_test.sh ${work_path}

cd ${work_path}
echo "*******examples_test ip_adapter_test begin***********"
(bash ip_adapter_test.sh) 2>&1 | tee ${log_dir}/ip_adapter_test.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "examples_test ip_adapter_test run success" >>"${log_dir}/ce_res.log"
else
    echo "examples_test ip_adapter_test run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******examples_test ip_adapter_test end***********"

cd ${cur_path}
echo ${cur_path}
work_path=${root_path}/PaddleMIX/ppdiffusers/examples/consistency_distillation/lcm_trainer
echo ${work_path}
/bin/cp -rf lcm_trainer_test.sh ${work_path}

cd ${work_path}
echo "*******examples_test lcm_trainer_test begin***********"
(bash lcm_trainer_test.sh) 2>&1 | tee ${log_dir}/lcm_trainer_test.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "examples_test lcm_trainer_test run success" >>"${log_dir}/ce_res.log"
else
    echo "examples_test lcm_trainer_test run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******examples_test lcm_trainer_test end***********"

# # 查看结果
cat ${log_dir}/ce_res.log
echo exit_code:${exit_code}
exit ${exit_code}