#!/bin/bash

echo "mix version:" >> ${log_dir}/res.log
cd ${root_path}/PaddleMIX
echo "mix branch:" >> ${log_dir}/res.log
git branch >> ${log_dir}/res.log
echo "mix commit:" >> ${log_dir}/res.log
git rev-parse HEAD >> ${log_dir}/res.log
cd ${root_path}

log_dir=${root_path}/log
echo "paddle version:" >> ${log_dir}/res.log
echo "mix commit:" >> ${log_dir}/res.log
python -c "import paddle;print(paddle.__git_commit__)" >> ${log_dir}/res.log