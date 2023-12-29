#!/bin/bash

log_dir=${root_path}/log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

echo "mix version:" >>${log_dir}/ce_res.log
cd ${root_path}/PaddleMIX
echo "mix branch:" >>${log_dir}/ce_res.log
git branch >>${log_dir}/ce_res.log
echo "mix commit:" >>${log_dir}/ce_res.log
git rev-parse HEAD >>${log_dir}/ce_res.log
cd ${root_path}

log_dir=${root_path}/log
echo "paddle version:" >>${log_dir}/ce_res.log
echo "paddle commit:" >>${log_dir}/ce_res.log
python -c "import paddle;print(paddle.__git_commit__)" >>${log_dir}/ce_res.log
