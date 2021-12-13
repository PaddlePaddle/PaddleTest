#! /bin/bash


test_mode=${TIPC_MODE:-lite_train_lite_infer}
test_mode=$(echo $test_mode | tr "," "\n")

cat full_chain_list_clas_unrun | while read config_file
do
# for config_file in `find . -name "*train_infer_python.txt" | grep -v "LeViT_384"  | grep -v "DeiT_base_patch16_384"`; do
    for mode in $test_mode; do
        mode=$(echo $mode | xargs)
        echo "step now"
        echo "======="$config_file"==========="
        echo "======="$mode"==========="
        bash test_tipc/prepare.sh $config_file $mode
        bash test_tipc/test_train_inference_python.sh $config_file $mode
    done
done
