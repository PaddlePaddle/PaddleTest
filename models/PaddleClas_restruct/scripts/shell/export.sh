# 输入变量：yaml、设置卡数CPU/SET_CUDA/SET_MULTI_CUDA

cd ${Project_path} #确定下执行路径
ls
ls ${Project_path}/../  #通过相对路径找到 scripts 的路径，需要想一个更好的方法替代
ls ${Project_path}/../scripts
cp ${Project_path}/../scripts/shell/prepare.sh .
source prepare.sh
bash prepare.sh ${1} ${2}

if [[ -d ${output_dir}/${model_name} ]];then
    params_dir=$(ls ${output_dir}/${model_name})
    echo "######  params_dir"
    echo $params_dir
    if [[ -f ${output_dir}/$params_dir/latest.pdparams ]];then
        pretrained_model=${output_dir}/$params_dir/latest
    else
        pretrained_model="null"
    fi
else
    pretrained_model="null"
fi

# export_model
if [[ ${1} =~ 'amp' ]];then
    python tools/export_model.py -c ${1} \
        -o Global.pretrained_model=${pretrained_model} \
        -o Global.save_inference_dir=./inference/${model_name} \
        -o Arch.data_format="NCHW" \
        > ${log_path}/export_model/${model_name}.log 2>&1
else
    python tools/export_model.py -c ${1} \
        -o Global.pretrained_model=${pretrained_model} \
        -o Global.save_inference_dir=./inference/${model_name} \
        > ${log_path}/export_model/${model_name}.log 2>&1
fi

if [[ $? -eq 0 ]] && [[ $(grep -c  "Error" ${log_path}/export_model/${model_name}.log) -eq 0 ]];then
    echo -e "\033[33m export_model of ${model_name}  successfully!\033[0m"| tee -a ${log_path}/result.log
    echo "export_exit_code: 0.0" >> ${log_path}/export_model/${model_name}.log
else
    cat ${log_path}/export_model/${model_name}.log
    echo -e "\033[31m export_model of ${model_name} failed!\033[0m" | tee -a ${log_path}/result.log
    echo "export_exit_code: 1.0" >> ${log_path}/export_model/${model_name}.log
fi
