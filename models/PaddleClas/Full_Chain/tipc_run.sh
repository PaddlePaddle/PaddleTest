#! /bin/bash


test_mode=${TIPC_MODE:-lite_train_lite_infer}
test_mode=$(echo $test_mode | tr "," "\n")

echo "grep rules"
echo ${grep_models}
echo ${grep_v_models}

find . -name "*train_infer_python.txt" > full_chain_list_clas_all_tmp
if [[ ${grep_models} =~ "undefined" ]]; then
    cat full_chain_list_clas_all_tmp | sort | uniq |grep -v ${grep_v_models} > full_chain_list_clas_all  #去重复
else
    cat full_chain_list_clas_all_tmp | sort | uniq |grep -v ${grep_v_models} |grep ${grep_models} > full_chain_list_clas_all  #去重复
fi

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
        bash -x upload.sh ${config_file} ${mode} || echo "upload model error on"`pwd`
        sleep 3
    done

end=`date +%s`
time=`echo $start $end | awk '{print $2-$1}'`
echo "${config_file} spend time seconds ${time}"
done

# update model_url latest
if [ -f "tipc_models_url_${REPO}.txt" ];then
    date_stamp=`date +%m_%d`
    push_file=./bce-python-sdk-0.8.27/BosClient.py
    cp "tipc_models_url_${REPO}.txt" "tipc_models_url_${REPO}_latest.txt"
    cp "tipc_models_url_${REPO}.txt" "tipc_models_url_${REPO}_${date_stamp}.txt"
    python2 ${push_file} "tipc_models_url_${REPO}_latest.txt" paddle-qa/fullchain_ce_test/model_download_link
    python2 ${push_file} "tipc_models_url_${REPO}_${date_stamp}.txt" paddle-qa/fullchain_ce_test/model_download_link
fi
