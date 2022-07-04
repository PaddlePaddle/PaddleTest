function export() {

    if [[ -f output/$params_dir/latest.pdparams ]];then
        local pretrained_model=output/$params_dir/latest
    else
        local pretrained_model=null #直接执行eval，下载预训练模型

    # export_model 
    if [[ ${line} =~ 'amp' ]];then
        python tools/export_model.py -c $line \
            -o Global.pretrained_model=${pretrained_model} \
            -o Global.save_inference_dir=./inference/${model} \
            -o Arch.data_format="NCHW" \
            > $log_path/export_model/${model}.log 2>&1
    else
        python tools/export_model.py -c $line \
            -o Global.pretrained_model=${pretrained_model} \
            -o Global.save_inference_dir=./inference/${model} \
            > $log_path/export_model/${model}.log 2>&1
    fi

    if [[ $? -eq 0 ]] && [[ $(grep -c  "Error" $log_path/export_model/${model}.log) -eq 0 ]];then
        echo -e "\033[33m export_model of ${model}  successfully!\033[0m"| tee -a $log_path/result.log
        echo "export_exit_code: 0.0" >> $log_path/export_model/${model}.log
    else
        cat $log_path/export_model/${model}.log
        echo -e "\033[31m export_model of ${model} failed!\033[0m" | tee -a $log_path/result.log
        echo "export_exit_code: 1.0" >> $log_path/export_model/${model}.log
    fi

}