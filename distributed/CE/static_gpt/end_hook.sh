#!/bin/bash
set -x
echo "end_hook"
cd static_gpt/manual_auto_parallel_model/gpt/
log_dir=$PWD/output

#hostname=`echo $(hostname)|awk -F '.baidu.com' '{print $1}'`
if [[ -e ${log_dir} ]]; then
    ls -a ${log_dir}
    auto_list=("auto_dp2" "auto_mp2" "auto_dp2mp2")
    #hybrud_list=("dp2" "mp2" "dp2mp2")
    msg_dp2=$(cat ${log_dir}/${auto_list[0]}/workerlog.0 | grep "step: 20000, loss: ")
    msg_mp2=$(cat ${log_dir}/${auto_list[1]}/workerlog.0 | grep "step: 20000, loss: ")
    msg_dp2mp2=$(cat ${log_dir}/${auto_list[2]}/workerlog.0 | grep "step: 20000, loss: ")
else
    echo "not found log dir: ${log_dir}"
fi
echo ${msg_dp2:19} ${msg_mp2:19} ${msg_dp2mp2:19}
python3 send_email.py ${msg_dp2:19} ${msg_mp2:19} ${msg_dp2mp2:19}
