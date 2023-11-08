#!/bin/bash

export http_proxy=${proxy};
export https_proxy=${proxy};
bash prepare.sh
unset http_proxy
unset https_proxy

work_path=`pwd`
echo ${work_path}

exit_code=0

cd ${work_path}

cd ${work_path}/paddlemix/
bash fire.sh
exit_code=$(($exit_code + $?))

cd ${work_path}/ppdiffusers/
bash fire.sh
exit_code=$(($exit_code + $?))

cd ${work_path}/ut/
bash fire.sh
exit_code=$(($exit_code + $?))

log_dir=${root_path}/log
cat ${log_dir}/res.log

echo exit_code:${exit_code}
exit ${exit_code}
exit ${exit_code}