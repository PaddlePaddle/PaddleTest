#!/usr/bin/env bash

cur_path=`pwd`
PY_CMD=$1

for module in $(cat CI_model_list.txt):
do
$PY_CMD test_${module}.py 2>&1 | tee -a results.log
done

num=`cat $cur_path/results.log | grep "failed" | wc -l`
if [ "${num}" -gt "0" ];then
exit 1
else
exit 0
fi
