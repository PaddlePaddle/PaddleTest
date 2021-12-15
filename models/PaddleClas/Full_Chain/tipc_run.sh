#! /bin/bash


test_mode=${TIPC_MODE:-lite_train_lite_infer}
test_mode=$(echo $test_mode | tr "," "\n")

find . -name "*train_infer_python.txt" > full_chain_list_clas_all_tmp
cat full_chain_list_clas_all_tmp | sort | uniq |grep -v PPLCNet_x0_75 > full_chain_list_clas_all  #去重复

cat full_chain_list_clas_all | while read config_file #手动定义
do

# for config_file in `find . -name "*train_infer_python.txt"`; do
start=`date +%s`
    for mode in $test_mode; do
        mode=$(echo $mode | xargs)
        echo "step now"
        echo "======="$config_file"==========="
        echo "======="$mode"==========="
        bash test_tipc/prepare.sh $config_file $mode
        bash test_tipc/test_train_inference_python.sh $config_file $mode
        sleep 3
    done

end=`date +%s`
time=`echo $start $end | awk '{print $2-$1}'`
echo "${config_file} spend time seconds ${time}"
done
