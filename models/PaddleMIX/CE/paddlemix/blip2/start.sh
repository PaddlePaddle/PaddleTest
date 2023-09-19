#!/bin/bash

cur_path=`pwd`
echo ${cur_path}


work_path=${root_path}/PaddleMIX/
echo ${work_path}

log_dir=${root_path}/log

# 检查上一级目录中是否存在log目录
if [ ! -d "$log_dir" ]; then
    # 如果log目录不存在，则创建它
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}/
exit_code=0

# 下载依赖、数据集和权重
bash prepare.sh

cd ${work_path}

# 训练
bash single_train.sh
bash multi_train.sh
exit_code=$(($exit_code + $?))
# 评估
bash single_eval.sh
bash multi_eval.sh
exit_code=$(($exit_code + $?))
# 预测
bash single_predict.sh
bash multi_predict.sh
exit_code=$(($exit_code + $?))

# # 查看结果
# cat ${log_dir}/ce_res.log

echo exit_code:${exit_code}
exit ${exit_code}