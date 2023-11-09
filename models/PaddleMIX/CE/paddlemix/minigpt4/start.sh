#!/bin/bash

cur_path=`pwd`
echo ${cur_path}


work_path=${root_path}/PaddleMIX/paddlemix/examples/minigpt4/
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

export http_proxy=${proxy}
export https_proxy=${proxy}
wget https://user-images.githubusercontent.com/35913314/242832479-d8070644-4713-465d-9c7e-9585024c1819.png
mv 242832479-d8070644-4713-465d-9c7e-9585024c1819.png example.png
unset http_proxy
unset https_proxy

bash minigpt4_7b.sh
exit_code=$(($exit_code + $?))
bash minigpt4_13b.sh
exit_code=$(($exit_code + $?))


# # 查看结果
# cat ${log_dir}/ce_res.log

echo exit_code:${exit_code}
exit ${exit_code}