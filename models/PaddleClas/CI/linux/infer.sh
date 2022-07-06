function infer() {

    #需要考虑如何跳过
        # reid
        # 识别方向
        # deephash
        # adaface_ir18

    if [[ -f output/$params_dir/latest.pdparams ]];then
        local pretrained_model=output/$params_dir/latest
    else
        local pretrained_model=null #直接执行eval，下载预训练模型

    python tools/infer.py -c $line \
        -o Global.pretrained_model=${pretrained_model} \
        > $log_path/infer/${model}.log 2>&1

    if [[ $? -eq 0 ]] && [[ $(grep -c  "Error" $log_path/infer/${model}.log) -eq 0 ]];then
        echo -e "\033[33m infer of ${model}  successfully!\033[0m"| tee -a $log_path/result.log
        echo "infer_exit_code: 0.0" >> $log_path/infer/${model}.log
    else
        cat $log_path/infer/${model}.log
        echo -e "\033[31m infer of ${model} failed!\033[0m"| tee -a $log_path/result.log
        echo "infer_exit_code: 1.0" >> $log_path/infer/${model}.log
    fi

}
