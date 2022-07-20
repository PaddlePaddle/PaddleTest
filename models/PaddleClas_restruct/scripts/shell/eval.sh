# 输入变量：yaml、设置卡数CPU/SET_CUDA/SET_MULTI_CUDA

cd ${Project_path} #确定下执行路径
ls
ls ${Project_path}/../  #通过相对路径找到 scripts 的路径，需要想一个更好的方法替代
ls ${Project_path}/../scripts
cp ${Project_path}/../scripts/shell/prepare.sh .
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
        pretrained_model="null"
    fi
else
    pretrained_model="null"
fi

if [[ ${1} =~ 'MV3_Large_1x_Aliproduct_DLBHC' ]] ; then
    python ${multi_flag} tools/eval.py -c $line \
        -o Global.pretrained_model=${pretrained_model} \
        -o DataLoader.Eval.sampler.batch_size=64 \
        > ${log_path}/eval/${model_name}.log 2>&1
else
    python ${multi_flag} tools/eval.py -c $line \
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
