#!/bin/bash

cur_path=`pwd`
echo ${cur_path}


work_path=${root_path}/PaddleMIX/tests/appflow/
echo ${work_path}

log_dir=${root_path}/log

# 检查上一级目录中是否存在log目录
if [ ! -d "$log_dir" ]; then
    # 如果log目录不存在，则创建它
    mkdir -p "$log_dir"
fi

exit_code=0

cd ${work_path}


echo "*******tests/appflow/ test_cviw begin***********"
(python test_cviw.py) 2>&1 | tee ${log_dir}/appflow_test_cviw.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ut_res.log
    echo "tests/appflow/ test_cviw run success" >> "${log_dir}/res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ut_res.log
    echo "tests/appflow/ test_cviw run fail" >> "${log_dir}/res.log"
fi
echo "*******tests/appflow/ test_cviw end***********"

echo exit_code:${exit_code}
exit ${exit_code}