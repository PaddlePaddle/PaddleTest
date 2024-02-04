#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/examples/consistency_distillation/lcm_trainer
echo ${work_path}

log_dir=${root_path}/log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}

cd ${work_path}
exit_code=0

# 下载依赖、数据集和权重
bash prepare.sh

# 单机训练
echo "*******ppdiffusers consistency_distillation/lcm_trainer singe_train begin***********"
(bash singe_train.sh) 2>&1 | tee ${log_dir}/consistency_distillation_lcm_trainer_singe_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers consistency_distillation/lcm_trainer singe_train run success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers consistency_distillation/lcm_trainer singe_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers consistency_distillation/lcm_trainer singe_train end***********"

echo "*******ppdiffusers consistency_distillation/lcm_trainer sdxl_singe_train begin***********"
(bash singe_train_sdxl.sh) 2>&1 | tee ${log_dir}/consistency_distillation_lcm_trainer_sdxl_singe_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers consistency_distillation/lcm_trainer sdxl_singe_train run success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers consistency_distillation/lcm_trainer sdxl_singe_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers consistency_distillation/lcm_trainer sdxl_singe_train end***********"


# 多机训练
echo "*******ppdiffusers consistency_distillation/lcm_trainer muti_train begin***********"
(bash muti_train.sh) 2>&1 | tee ${log_dir}/consistency_distillation_lcm_trainer_muti_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers consistency_distillation/lcm_trainer_muti_train run success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers consistency_distillation/lcm_trainer_muti_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******consistency_distillation/lcm_trainer muti_train end***********"

echo "*******ppdiffusers consistency_distillation/lcm_trainer sdxl_multi_train begin***********"
(bash muti_train_sdxl.sh) 2>&1 | tee ${log_dir}/consistency_distillation_lcm_trainer_sdxl_multi_train.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "ppdiffusers consistency_distillation/lcm_trainer sdxl_multi_train run success" >>"${log_dir}/ce_res.log"
else
    echo "ppdiffusers consistency_distillation/lcm_trainer sdxl_multi_train run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******ppdiffusers consistency_distillation/lcm_trainer sdxl_multi_train end***********"


# # 查看结果
# cat ${log_dir}/ce_res.log
rm -rf ${work_path}/lcm_lora_*
rm -rf ${work_path}/lcm_sdxl_*
rm -rf ${work_path}/data/
rm -rf ${work_path}/image_*

echo exit_code:${exit_code}
exit ${exit_code}
