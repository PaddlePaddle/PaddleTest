# 输入变量：yaml、设置卡数CPU/SET_CUDA/SET_MULTI_CUDA

cd ${Project_path} #确定下执行路径
cp ${Project_path}/../scripts/shell/prepare.sh . # #通过相对路径找到 scripts 的路径，需要想一个更好的方法替代
source prepare.sh
bash prepare.sh ${1} ${2}

# 廷权临时增加规则 220413
if [[ ${1} =~ 'ultra' ]];then
    cp ${1} ${1}_tmp
    sed -i '/output_fp16: True/d' ${1}
fi

if [[ -d ${output_dir}/${model_name} ]];then
    params_dir=$(ls ${output_dir}/${model_name})
    echo "######  params_dir"
    echo $params_dir
    if [[ -f ${output_dir}/${model_name}/$params_dir/latest.pdparams ]];then
        pretrained_model=${output_dir}/${model_name}/$params_dir/latest
    else
        pretrained_model="null"  #使用初始化参数评估
    fi
else
    pretrained_model="null"   #使用预训练模型评估 单独写一个use_pretrain在里面按逻辑获取预训练模型
fi

if [[ ${1} =~ 'MV3_Large_1x_Aliproduct_DLBHC' ]] ; then
    python ${multi_flag} tools/eval.py -c ${1} \
        -o Global.pretrained_model=${pretrained_model} \
        -o DataLoader.Eval.sampler.batch_size=64 \
        > ${log_path}/eval/${model_name}.log 2>&1
else
    python ${multi_flag} tools/eval.py -c ${1} \
        -o Global.pretrained_model=${pretrained_model} \
        > ${log_path}/eval/${model_name}.log 2>&1
fi

# 廷权临时增加规则 220413
if [[ ${1} =~ 'ultra' ]];then
    rm -rf ${1}
    mv ${1}_tmp ${1}
fi

if [[ $? -eq 0 ]] && [[ $(grep -c  "Error" ${log_path}/eval/${model_name}.log) -eq 0 ]];then
    echo -e "\033[33m eval of ${model_name}  successfully!\033[0m"| tee -a ${log_path}/result.log
    echo "eval_exit_code: 0.0" >> ${log_path}/eval/${model_name}.log
else
    cat ${log_path}/eval/${model_name}.log
    echo -e "\033[31m eval of ${model_name} failed!\033[0m" | tee -a ${log_path}/result.log
    echo "eval_exit_code: 1.0" >> ${log_path}/eval/${model_name}.log
fi
