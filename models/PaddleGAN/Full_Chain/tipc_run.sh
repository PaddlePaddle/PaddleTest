#! /bin/bash


test_mode=${TIPC_MODE:-lite_train_lite_infer,whole_infer,lite_train_whole_infer,whole_train_whole_infer}
test_mode=$(echo $test_mode | tr "," "\n")

for config_file in `find . -name "*train_infer_python.txt"`; do
    for mode in $test_mode; do
        mode=$(echo $mode | xargs)
        bash test_tipc/prepare.sh $config_file $mode
        bash test_tipc/test_train_inference_python.sh $config_file $mode
    done
done