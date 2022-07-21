# 输入变量：yaml、设置卡数CPU/SET_CUDA/SET_MULTI_CUDA 、trained/pretrained

cd ${Project_path} #确定下执行路径
\cp -r -f ${Project_path}/../scripts/shell/prepare.sh . # #通过相对路径找到 scripts 的路径，需要想一个更好的方法替代
\cp -r -f ${Project_path}/../scripts/shell/choose_model.sh .

source prepare.sh
export output_type=${3:-trained} #作为参数传入
# arr=("trained" "pretrained")
# for output_type in ${arr[@]}
# do
source choose_model.sh

case ${model_type} in
ImageNet|slim|metric_learning)
    python tools/infer.py -c ${1} \
        -o Global.pretrained_model=${pretrained_model} \
        -o Global.output_dir=${output_dir}/${model_name} \
        > ${log_path}/infer/${model_name}_${output_type}.log 2>&1
;;
Cartoonface)
    echo "infer_exit_code: unspported" >> ${log_path}/infer/${model_name}_${output_type}.log
;;
DeepHash|GeneralRecognition)
    echo "infer_exit_code: unspported" >> ${log_path}/infer/${model_name}_${output_type}.log
;;
Logo)
    echo "infer_exit_code: unspported" >> ${log_path}/infer/${model_name}_${output_type}.log
;;
Products)
    echo "infer_exit_code: unspported" >> ${log_path}/infer/${model_name}_${output_type}.log
;;
PULC)
    echo "infer_exit_code: unspported" >> ${log_path}/infer/${model_name}_${output_type}.log
;;
reid)
    echo "infer_exit_code: unspported" >> ${log_path}/infer/${model_name}_${output_type}.log
;;
Vehicle)
    echo "infer_exit_code: unspported" >> ${log_path}/infer/${model_name}_${output_type}.log
;;
esac

if [[ $? -eq 0 ]] && [[ $(grep -c  "Error" ${log_path}/infer/${model_name}_${output_type}.log) -eq 0 ]];then
    echo -e "\033[33m infer of ${model_name}_${output_type}  successfully!\033[0m"| tee -a ${log_path}/result.log
    echo "infer_exit_code: 0.0" >> ${log_path}/infer/${model_name}_${output_type}.log
else
    cat ${log_path}/infer/${model_name}_${output_type}.log
    echo -e "\033[31m infer of ${model_name}_${output_type} failed!\033[0m"| tee -a ${log_path}/result.log
    echo "infer_exit_code: 1.0" >> ${log_path}/infer/${model_name}_${output_type}.log
fi

# done
