
# 输入变量：yaml、设置卡数 CPU SET_CUDA SET_MULTI_CUDA 、训练的模型动态图/静态图/收敛性( dynamic static convergence )

cd ${Project_path} #确定下执行路径
# ls
# ls ${Project_path}/../  #通过相对路径找到 scripts 的路径，需要想一个更好的方法替代
# ls ${Project_path}/../scripts
\cp -r -f ${Project_path}/../scripts/shell/prepare.sh .
source prepare.sh

#区分动态图、静态图
if [[ ${3} =~ "dynamic" ]] || [[ ${3} =~ "convergence" ]];then
    export train_type="tools/train.py"
else
    export train_type="ppcls/static/train.py"
fi

case ${3} in #动态图/静态图/收敛性
dynamic|static)
    common_par="-o Global.epochs=2 \
    -o Global.save_interval=2 \
    -o Global.eval_interval=2 \
    -o Global.seed=1234 \
    -o DataLoader.Train.loader.num_workers=0 \
    -o DataLoader.Train.sampler.shuffle=False  \
    -o Global.output_dir=${output_dir}/${model_name} \
    -o Global.device=${set_cuda_device}"
    if [[ ${1} =~ 'GeneralRecognition' ]]; then
        python ${multi_flag} ${train_type} -c ${1} \
            -o DataLoader.Train.sampler.batch_size=32 \
            -o DataLoader.Train.dataset.image_root=./dataset/Inshop/ \
            -o DataLoader.Train.dataset.cls_label_path=./dataset/Inshop/train_list.txt \
            ${common_par} > ${log_path}/train/${model_name}_${card}.log 2>&1
    elif [[ ${1} =~ 'MV3_Large_1x_Aliproduct_DLBHC' ]] ; then
        python ${multi_flag} ${train_type} -c ${1} \
            -o DataLoader.Eval.sampler.batch_size=64 \
            -o DataLoader.Train.sampler.batch_size=64 \
            -o DataLoader.Train.dataset.image_root=./dataset/Inshop/ \
            -o DataLoader.Train.dataset.cls_label_path=./dataset/Inshop/train_list.txt \
            ${common_par} > ${log_path}/train/${model_name}_${card}.log 2>&1
    elif [[ ${1} =~ 'quantization' ]] ; then
        python ${multi_flag} ${train_type} -c ${1} \
            -o DataLoader.Train.sampler.batch_size=32 \
            ${common_par} > ${log_path}/train/${model_name}_${card}.log 2>&1
    else
        python ${multi_flag} tools/train.py -c ${1}  \
            ${common_par} > ${log_path}/train/${model_name}_${card}.log 2>&1
    fi
    params_dir=$(ls ${output_dir}/${model_name})
    echo "######  params_dir"
    echo $params_dir
    cat ${log_path}/train/${model_name}_${card}.log | grep "Memory Usage (MB)"

    if ([[ -f "${output_dir}/${model_name}/$params_dir/latest.pdparams" ]] \
        || [[ -f "${output_dir}/${model_name}/$params_dir/0/ppcls.pdmodel" ]]) && [[ $? -eq 0 ]] \
        && [[ $(grep -c  "Error" ${log_path}/train/${model_name}_${card}.log) -eq 0 ]];then
        echo -e "\033[33m training in ${card} of ${model_name}  successfully!\033[0m"|tee -a ${log_path}/result.log
        echo "training_multi_exit_code: 0.0" >> ${log_path}/train/${model_name}_${card}.log
    else
        cat ${log_path}/train/${model_name}_${card}.log
        echo -e "\033[31m training in ${card} of ${model_name} failed!\033[0m"|tee -a ${log_path}/result.log
        echo "training_multi_exit_code: 1.0" >> ${log_path}/train/${model_name}_${card}.log
    fi
;;
convergence)
    python ${multi_flag} tools/train.py -c ${1}  \
        -o Global.output_dir=${output_dir}/${model_name} \
        > ${log_path}/train/${model_name}_convergence.log 2>&1
    params_dir=$(ls ${output_dir}/${model_name})
    echo "######  params_dir"
    echo $params_dir
    cat ${log_path}/train/${model_name}_convergence.log | grep "Memory Usage (MB)"

    if ([[ -f "${output_dir}/${model_name}/$params_dir/latest.pdparams" ]] \
        || [[ -f "${output_dir}/${model_name}/$params_dir/0/ppcls.pdmodel" ]]) && [[ $? -eq 0 ]] \
        && [[ $(grep -c  "Error" ${log_path}/train/${model_name}_convergence.log) -eq 0 ]];then
        echo -e "\033[33m training in convergence of ${model_name}  successfully!\033[0m"|tee -a ${log_path}/result.log
        echo "training_multi_exit_code: 0.0" >> ${log_path}/train/${model_name}_convergence.log
    else
        cat ${log_path}/train/${model_name}_convergence.log
        echo -e "\033[31m training in convergence of ${model_name} failed!\033[0m"|tee -a ${log_path}/result.log
        echo "training_multi_exit_code: 1.0" >> ${log_path}/train/${model_name}_convergence.log
    fi
;;
esac
