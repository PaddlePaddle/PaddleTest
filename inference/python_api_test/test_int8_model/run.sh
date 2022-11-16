#! /bin/bash

bash prepare.sh
mv models models.bak

OLD_IFS="${IFS}"
IFS=","

for mode in $MODE
do
    echo ${mode}

    cp -r models.bak models
    if [[ ${mode} =~ "trt_int8" ]] || [[ ${mode} =~ "trt_fp16" ]]
    then
        bash run_${mode}.sh > eval_${mode}_acc.log.tmp 2>&1
    fi
    bash run_${mode}.sh > eval_${mode}_acc.log 2>&1
    rm -rf models
done

IFS="${OLD_IFS}"

python get_benchmark_info.py
