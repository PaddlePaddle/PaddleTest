# 输入变量：yaml、设置卡数CPU/SET_CUDA/SET_MULTI_CUDA、 trained/pretrained

cd ${Project_path} #确定下执行路径
\cp -r -f ${Project_path}/../scripts/shell/prepare.sh . # #通过相对路径找到 scripts 的路径，需要想一个更好的方法替代
\cp -r -f ${Project_path}/../scripts/shell/choose_model.sh .

source prepare.sh
export output_type=${3:-trained} #作为参数传入
# arr=("trained" "pretrained")
# for output_type in ${arr[@]}
# do
source choose_model.sh
if [[ ${output_type} == 'pretrained' ]];then
    export save_inference_dir=${params_dir}_infer
else
    export save_inference_dir=./inference/${model_name}
fi

# export_model
if [[ ${1} =~ 'amp' ]];then
    python tools/export_model.py -c ${1} \
        -o Global.pretrained_model=${pretrained_model} \
        -o Global.save_inference_dir=${save_inference_dir} \
        -o Arch.data_format="NCHW" \
        -o Global.output_dir=${output_dir}/${model_name} \
        > ${log_path}/export_model/${model_name}_${output_type}.log 2>&1
else
    python tools/export_model.py -c ${1} \
        -o Global.pretrained_model=${pretrained_model} \
        -o Global.save_inference_dir=${save_inference_dir} \
        -o Global.output_dir=${output_dir}/${model_name} \
        > ${log_path}/export_model/${model_name}_${output_type}.log 2>&1
fi

if [[ $? -eq 0 ]] && [[ $(grep -c  "Error" ${log_path}/export_model/${model_name}_${output_type}.log) -eq 0 ]];then
    echo -e "\033[33m export_model of ${model_name}_${output_type}  successfully!\033[0m"| tee -a ${log_path}/result.log
    echo "export_exit_code: 0.0" >> ${log_path}/export_model/${model_name}_${output_type}.log
else
    cat ${log_path}/export_model/${model_name}_${output_type}.log
    echo -e "\033[31m export_model of ${model_name}_${output_type} failed!\033[0m" | tee -a ${log_path}/result.log
    echo "export_exit_code: 1.0" >> ${log_path}/export_model/${model_name}_${output_type}.log
fi

# done
