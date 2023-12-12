#!/bin/bash

export PPNLP_HOME=/home/cache_weight
export PPMIX_HOME=/home/cache_weight

cur_path=`pwd`
echo ${cur_path}

work_path=${root_path}/PaddleMIX/tests/appflow
echo ${work_path}

log_dir=${root_path}/log


if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi


/bin/cp -rf ./* ${work_path}

cd ${work_path}

exit_code=0

echo "*******paddlemix appflow test_cviw begin***********"
(python test_cviw.py) 2>&1 | tee ${log_dir}/test_cviw.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix appflow test_cviw success" >> "${log_dir}/ce_res.log"
else
    echo "paddlemix appflow test_cviw fail" >> "${log_dir}/ce_res.log"
fi
echo "*******paddlemix appflow test_cviw end***********"

echo "*******paddlemix appflow test_inpainting begin***********"
(python test_inpainting.py) 2>&1 | tee ${log_dir}/test_inpainting.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix appflow test_inpainting success" >> "${log_dir}/ce_res.log"
else
    echo "paddlemix appflow test_inpainting fail" >> "${log_dir}/ce_res.log"
fi
echo "*******paddlemix appflow test_inpainting end***********"

export RUN_SLOW_TEST=True
echo "*******export RUN_SLOW_TEST=True paddlemix appflow test_cviw begin***********"
(python test_cviw.py) 2>&1 | tee ${log_dir}/test_cviw_2.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "export RUN_SLOW_TEST=True paddlemix appflow test_cviw success" >> "${log_dir}/ce_res.log"
else
    echo "export RUN_SLOW_TEST=True paddlemix appflow test_cviw fail" >> "${log_dir}/ce_res.log"
fi
echo "*******export RUN_SLOW_TEST=True paddlemix appflow test_cviw end***********"
unset RUN_SLOW_TEST

echo exit_code:${exit_code}
exit ${exit_code}