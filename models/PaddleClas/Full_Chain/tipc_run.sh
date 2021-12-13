#! /bin/bash


test_mode=${TIPC_MODE:-lite_train_lite_infer}
test_mode=$(echo $test_mode | tr "," "\n")

cat full_chain_list_clas_all | while read config_file
do
start=`date +%s`

# for config_file in `find . -name "*train_infer_python.txt"`; do
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
