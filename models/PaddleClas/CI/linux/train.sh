function train() {  #line log_path model cudaid2

    export CUDA_VISIBLE_DEVICES=${4}  #区分卡数
    if [[ ${4} =~ "," ]];then
        local card=2card
    else
        local card=1card
    fi

    elif [[ ${1} =~ 'GeneralRecognition' ]]; then
        python -m paddle.distributed.launch tools/train.py  -c ${1} \
            -o Global.epochs=2 \
            -o Global.save_interval=2 \
            -o Global.eval_interval=2 \
            -o Global.seed=1234 \
            -o DataLoader.Train.loader.num_workers=0 \
            -o DataLoader.Train.sampler.shuffle=False  \
            -o DataLoader.Train.sampler.batch_size=32 \
            -o DataLoader.Train.dataset.image_root=./dataset/Inshop/ \
            -o DataLoader.Train.dataset.cls_label_path=./dataset/Inshop/train_list.txt \
            -o Global.output_dir=output \
            > ${2}/train/${3}_${card}.log 2>&1
    elif [[ ${1} =~ 'MV3_Large_1x_Aliproduct_DLBHC' ]] ; then
        python -m paddle.distributed.launch tools/train.py  -c ${1} \
            -o Global.epochs=2 \
            -o Global.save_interval=2 \
            -o Global.eval_interval=2 \
            -o Global.seed=1234 \
            -o DataLoader.Train.loader.num_workers=0 \
            -o DataLoader.Train.sampler.shuffle=False  \
            -o DataLoader.Eval.sampler.batch_size=64 \
            -o DataLoader.Train.sampler.batch_size=64 \
            -o DataLoader.Train.dataset.image_root=./dataset/Inshop/ \
            -o DataLoader.Train.dataset.cls_label_path=./dataset/Inshop/train_list.txt \
            -o Global.output_dir=output \
            > ${2}/train/${3}_${card}.log 2>&1
    elif [[ ${1} =~ 'quantization' ]] ; then
        python -m paddle.distributed.launch tools/train.py  -c ${1} \
            -o Global.epochs=2 \
            -o Global.save_interval=2 \
            -o Global.eval_interval=2 \
            -o Global.seed=1234 \
            -o DataLoader.Train.loader.num_workers=0 \
            -o DataLoader.Train.sampler.shuffle=False  \
            -o DataLoader.Train.sampler.batch_size=32 \
            -o Global.output_dir=output \
            > ${2}/train/${3}_${card}.log 2>&1
    else
        python -m paddle.distributed.launch tools/train.py -c ${1}  \
            -o Global.epochs=2  \
            -o Global.seed=1234 \
            -o Global.output_dir=output \
            -o DataLoader.Train.loader.num_workers=0 \
            -o DataLoader.Train.sampler.shuffle=False  \
            -o Global.eval_interval=2  \
            -o Global.save_interval=2 \
            > ${2}/train/${3}_${card}.log 2>&1
    fi
    params_dir=$(ls output)
    echo "######  params_dir"
    echo $params_dir
    cat $log_path/train/${model}_2card.log | grep "Memory Usage (MB)"

    if ([[ -f "output/$params_dir/latest.pdparams" ]] || [[ -f "output/$params_dir/0/ppcls.pdmodel" ]]) && [[ $? -eq 0 ]] \
        && [[ $(grep -c  "Error" $log_path/train/${model}_2card.log) -eq 0 ]];then
        echo -e "\033[33m training multi of ${model}  successfully!\033[0m"|tee -a $log_path/result.log
        echo "training_multi_exit_code: 0.0" >> $log_path/train/${model}_2card.log
    else
        cat $log_path/train/${model}_2card.log
        echo -e "\033[31m training multi of ${model} failed!\033[0m"|tee -a $log_path/result.log
        echo "training_multi_exit_code: 1.0" >> $log_path/train/${model}_2card.log
    fi
}

function train_static() {   #line log_path model cudaid2

    export CUDA_VISIBLE_DEVICES=${4}  #区分卡数
    if [[ ${4} =~ "," ]];then
        local card=2card
    else
        local card=1card
    fi

    python -m paddle.distributed.launch ppcls/static/train.py  \
        -c ${1} -o Global.epochs=1 \
        -o Global.seed=1234 \
        -o DataLoader.Train.loader.num_workers=0 \
        -o DataLoader.Train.sampler.shuffle=False  \
        -o Global.output_dir=output \
        > ${2}/train/${3}_${card}.log 2>&1
    params_dir=$(ls output)
    echo "######  params_dir"
    echo $params_dir
    cat $log_path/train/${model}_2card.log | grep "Memory Usage (MB)"

    if ([[ -f "output/$params_dir/latest.pdparams" ]] || [[ -f "output/$params_dir/0/ppcls.pdmodel" ]]) && [[ $? -eq 0 ]] \
        && [[ $(grep -c  "Error" $log_path/train/${model}_2card.log) -eq 0 ]];then
        echo -e "\033[33m training multi of ${model}  successfully!\033[0m"|tee -a $log_path/result.log
        echo "training_multi_exit_code: 0.0" >> $log_path/train/${model}_2card.log
    else
        cat $log_path/train/${model}_2card.log
        echo -e "\033[31m training multi of ${model} failed!\033[0m"|tee -a $log_path/result.log
        echo "training_multi_exit_code: 1.0" >> $log_path/train/${model}_2card.log
    fi
}

function train_convergence() {   #line log_path model cudaid2

    export CUDA_VISIBLE_DEVICES=${4}  #区分卡数
    if [[ ${4} =~ "," ]];then
        local card=multi_card
    else
        local card=1card
    fi

    python -m paddle.distributed.launch tools/train.py -c ${1}  \
        -o Global.output_dir=output \
        > ${2}/train/${3}_${card}.log 2>&1
    fi
    params_dir=$(ls output)
    echo "######  params_dir"
    echo $params_dir
    cat $log_path/train/${model}_2card.log | grep "Memory Usage (MB)"

    if ([[ -f "output/$params_dir/latest.pdparams" ]] || [[ -f "output/$params_dir/0/ppcls.pdmodel" ]]) && [[ $? -eq 0 ]] \
        && [[ $(grep -c  "Error" $log_path/train/${model}_2card.log) -eq 0 ]];then
        echo -e "\033[33m training multi of ${model}  successfully!\033[0m"|tee -a $log_path/result.log
        echo "training_multi_exit_code: 0.0" >> $log_path/train/${model}_2card.log
    else
        cat $log_path/train/${model}_2card.log
        echo -e "\033[31m training multi of ${model} failed!\033[0m"|tee -a $log_path/result.log
        echo "training_multi_exit_code: 1.0" >> $log_path/train/${model}_2card.log
    fi

}