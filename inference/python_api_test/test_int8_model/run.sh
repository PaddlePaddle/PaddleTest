#! /bin/bash

bash prepare.sh
mv models models.bak

OLD_IFS="${IFS}"
IFS=","

for mode in $MODE
do
    echo "==========START ${mode}========="

    cp -r models.bak models
    bash run_${mode}.sh > eval_${mode}_acc.log 2>&1
    rm -rf models
done

IFS="${OLD_IFS}"
