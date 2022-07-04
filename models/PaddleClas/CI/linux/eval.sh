function eval() { # 多卡eval 

    export CUDA_VISIBLE_DEVICES=${4}  #区分卡数
    if [[ ${4} =~ "," ]];then
        local card=2card
    else
        local card=1card
    fi

    if [[ ${line} =~ 'ultra' ]];then
        cp ${line} ${line}_tmp #220413 fix tingquan
        sed -i '/output_fp16: True/d' ${line}
    fi

    if [[ -f output/$params_dir/latest.pdparams ]];then
        local pretrained_model=output/$params_dir/latest
    else
        local pretrained_model=null #直接执行eval，下载预训练模型

    if [[ ${line} =~ 'MV3_Large_1x_Aliproduct_DLBHC' ]] ; then
        python -m paddle.distributed.launch tools/eval.py -c $line \
            -o Global.pretrained_model=${pretrained_model} \
            -o DataLoader.Eval.sampler.batch_size=64 \
            > $log_path/eval/${category}_${model}.log 2>&1
    else
        python -m paddle.distributed.launch tools/eval.py -c $line \
            -o Global.pretrained_model=${pretrained_model} \
            > $log_path/eval/${model}.log 2>&1
    fi
    
    if [[ ${line} =~ 'ultra' ]];then
        rm -rf ${line}
        mv ${line}_tmp ${line} #220413 fix tingquan
    fi

    if [[ $? -eq 0 ]] && [[ $(grep -c  "Error" $log_path/eval/${model}.log) -eq 0 ]];then
        echo -e "\033[33m eval of ${model}  successfully!\033[0m"| tee -a $log_path/result.log
        echo "eval_exit_code: 0.0" >> $log_path/eval/${model}.log
    else
        cat $log_path/eval/${model}.log
        echo -e "\033[31m eval of ${model} failed!\033[0m" | tee -a $log_path/result.log
        echo "eval_exit_code: 1.0" >> $log_path/eval/${model}.log
    fi
}
