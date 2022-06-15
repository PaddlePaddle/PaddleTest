#!/usr/bin/env bash
# 这里面的可以直接写训练case也可以跳转到script下（先将有差异的统一）
print_info(){
cat ${log_path}/$2.log
if [ $1 -ne 0 ];then
    echo "exit_code: 1.0" >> ${log_path}/$2.log
    #copy 一份失败的日志
    cp ${log_path}/$2.log ${log_path}/$2_FAIL.log
else
    echo "exit_code: 0.0" >> ${log_path}/$2.log
fi
}
# 1 waybill_ie
waybill_ie(){
# 固定接收5个参数，有扩展可以自定义
device=$1
system=$2
cudaid1=$3
cudaid2=$4
mode_tag=$5
step=$6 # 指定跑的阶段；为以后的case精细化做准备
# 这里怎么兼容cpu/gpu；linux、mac
if [[ ${mode_tag} == "CE" ]];then
    log_path=${log_path}/waybill_ie/
    default_list="train,python_infer"
else
    log_path=$log_path
    default_list="train"
fi
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
train(){
    # 定义train阶段
    cd ./waybill_ie
    bash data_proc.sh
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash train.sh ${device} ernie > $log_path/train_${device}_ernie.log 2>&1
    print_info $? train_${device}_ernie
    bash train.sh ${device} bigru_crf > $log_path/train_${device}_bigru_crf.log 2>&1
    print_info $? train_${device}_bigru_crf
    bash train.sh ${device} ernie_crf > $log_path/train_${device}_ernie_crf.log 2>&1
    print_info $? train_${device}_ernie_crf
    cd ..
}
python_infer(){
    cd ./waybill_ie
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash python_infer.sh ${device} ernie > ${log_path}/python_infer_ernie_${device}.log 2>&1
    print_info $? python_infer_ernie_${device}
    bash python_infer.sh ${device} ernie_crf > ${log_path}/python_infer_ernie_crf_${device}.log 2>&1
    print_info $? python_infer_ernie_crf_${device}
    bash python_infer.sh ${device} bigru_crf > ${log_path}/python_infer_bigru_crf_${device}.log 2>&1
    print_info $? python_infer_bigru_crf_${device}
    cd ..
}
if [[ ${step} && ${step} == "all" ]];then
    exec_list=(${default_list//,/ })
elif [[ ${step} ]];then
    exec_list=(${step//,/ })
else
    exec_list=(${default_list//,/ })
fi
echo ${exec_list[@]}
for case in ${exec_list[@]};do
    ${case}
done
}
# 2 msra_ner
msra_ner(){
# 固定接收5个参数，有扩展可以自定义
default_list="train,eval"
device=$1
system=$2
cudaid1=$3
cudaid2=$4
mode_tag=$5
step=$6
# 这里怎么兼容cpu/gpu；linux、mac
if [[ ${mode_tag} == "CE" ]];then
    log_path=${log_path}/msra_ner/
    cd ./msra_ner
    bash data_proc.sh
    cd ..
    MAX_STEPS=1000
    SAVE_STEPS=500
    LOGGING_STEPS=10
    MODEL_STEP=500
else
    log_path=$log_path
    MAX_STEPS=2
    SAVE_STEPS=2
    LOGGING_STEPS=1
    MODEL_STEP=2
fi
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
# 单卡训练
train(){
    cd ./msra_ner
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash train.sh ${device} single ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} $LOGGING_STEPS >$log_path/train_single_${device}.log 2>&1
    print_info $? train_single_${device}
    # 多卡训练
    unset CUDA_VISIBLE_DEVICES
    bash train.sh ${device} multi ${cudaid2} ${MAX_STEPS} ${SAVE_STEPS} $LOGGING_STEPS >$log_path/train_multi_${device}.log 2>&1
    print_info $? train_multi_${device}
    cd ..
}
## eval
eval(){
    cd ./msra_ner
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash infer.sh ${device} ${MODEL_STEP} > $log_path/infer_${device}.log 2>&1
    print_info $? infer_${device}
    bash eval.sh ${device} ${MODEL_STEP} > $log_path/eval_${device}.log 2>&1
    print_info $? eval_${device}
    cd ..
}
if [[ ${step} && ${step} == "all" ]];then
    exec_list=(${default_list//,/ })
elif [[ ${step} ]];then
    exec_list=(${step//,/ })
else
    exec_list=(${default_list//,/ })
fi
echo ${exec_list[@]}
for case in ${exec_list[@]};do
    ${case}
done
}
# 3 glue
glue() {
# 固定接收5个参数，有扩展可以自定义
default_list="train"
device=$1
system=$2
cudaid1=$3
cudaid2=$4
mode_tag=$5
step=$6
# 这里怎么兼容cpu/gpu；linux、mac
if [[ ${mode_tag} == "CE" ]];then
    log_path=${log_path}/glue/
    MAX_STEPS=10
    SAVE_STEPS=10
    LOGGING_STEPS=1
else
    log_path=$log_path
    MAX_STEPS=1
    SAVE_STEPS=1
    LOGGING_STEPS=1
fi
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
train(){
    cd ./glue
    # 单卡
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash train.sh ${device} 'single' 'bert' 'bert-base-uncased' 'SST-2' 1e-4 ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/train_bert-base-uncased_SST-2_single_${device}.log 2>&1
    print_info $? train_bert-base-uncased_SST-2_single_${device}
    #多卡
    export CUDA_VISIBLE_DEVICES=${cudaid2}
    bash train.sh ${device} 'multi' 'bert' 'bert-base-uncased' 'SST-2' 1e-4 ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/train_bert-base-uncased_SST-2_multi_${device}.log 2>&1
    print_info $? train_bert-base-uncased_SST-2_single_${device}
    cd ..
}
if [[ ${step} && ${step} == "all" ]];then
    exec_list=(${default_list//,/ })
elif [[ ${step} ]];then
    exec_list=(${step//,/ })
else
    exec_list=(${default_list//,/ })
fi
echo ${exec_list[@]}
for case in ${exec_list[@]};do
    ${case}
done
}

# 4 bert
# 5 skep (max save 不可控 内置)
# 6 bigbird
bigbird(){
# 固定接收5个参数，有扩展可以自定义
device=$1
system=$2
cudaid1=$3
cudaid2=$4
mode_tag=$5
step=$6
# 这里怎么兼容cpu/gpu；linux、mac
if [[ ${mode_tag} == "CE" ]];then
    log_path=${log_path}/bigbird/
    cd ./bigbird
    bash data_proc.sh
    cd ..
    MAX_STEPS=10
    SAVE_STEPS=10
    LOGGING_STEPS=1
    default_list="train,finetune"
else
    log_path=$log_path
    MAX_STEPS=1
    SAVE_STEPS=1
    LOGGING_STEPS=1
    default_list="train"
fi
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
train(){
    cd ./bigbird
    if [[ ${system} == "mac" ]];then
        bash train.sh ${device} 'bigbird-base-uncased' 'mac' 'None' ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/train_mac_${device}.log 2>&1
        print_info $? train_mac_${device}
    else
        unset CUDA_VISIBLE_DEVICES
        bash train.sh ${device} 'bigbird-base-uncased' 'single' ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/train_single_${device}.log 2>&1
        print_info $? train_single_${device}
        unset CUDA_VISIBLE_DEVICES
        bash train.sh ${device} 'bigbird-base-uncased' 'multi' ${cudaid2} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/train_multi_${device}.log 2>&1
        print_info $? train_multi_${device}
    fi
    cd ..
}
finetune(){
    cd ./bigbird
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash finetune.sh ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_${device}.log 2>&1
    print_info $? finetune_${device}
    cd ..
}
glue(){
    cd ./bigbird
    bash run_glue.sh ${device} ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/run_glue_${device}.log 2>&1
    print_info $? run_glue_${device}
    cd ..
}
if [[ ${step} && ${step} == "all" ]];then
    exec_list=(${default_list//,/ })
elif [[ ${step} ]];then
    exec_list=(${step//,/ })
else
    exec_list=(${default_list//,/ })
fi
echo ${exec_list[@]}
for case in ${exec_list[@]};do
    ${case}
done
}
# 7 electra
electra(){
# 固定接收5个参数，有扩展可以自定义
device=$1
system=$2
cudaid1=$3
cudaid2=$4
mode_tag=$5
step=$6
# 这里怎么兼容cpu/gpu；linux、mac
if [[ ${mode_tag} == "CE" ]];then
    log_path=${log_path}/electra/
    MAX_STEPS=12
    SAVE_STEPS=10
    LOGGING_STEPS=1
    default_list="train,finetune,infer"
else
    log_path=$log_path
    MAX_STEPS=1
    SAVE_STEPS=1
    LOGGING_STEPS=1
    default_list="train"
fi
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
train(){
    cd ./electra
    bash data_proc.sh
    # 单卡训练
    export DATA_DIR=./BookCorpus/
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash train.sh 'single' ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/single_card_train.log 2>&1
    print_info $? single_card_train
    export CUDA_VISIBLE_DEVICES=${cudaid2}
    bash train.sh 'multi' ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/multi_cards_train.log 2>&1
    print_info $? multi_cards_train
    cd ..
}
finetune(){
    cd ./electra
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash finetune.sh > $log_path/single_fine-tune.log 2>&1
    print_info $? single_fine-tune
    cd ..
}
infer(){
    cd ./electra
    bash infer.sh > $log_path/infer.log 2>&1
    print_info $? infer
    cd ..
}
if [[ ${step} && ${step} == "all" ]];then
    exec_list=(${default_list//,/ })
elif [[ ${step} ]];then
    exec_list=(${step//,/ })
else
    exec_list=(${default_list//,/ })
fi
echo ${exec_list[@]}
for case in ${exec_list[@]};do
    ${case}
done
}
# 9 ernie-1.0
ernie-1.0(){
# 固定接收5个参数，有扩展可以自定义
default_list="train"
device=$1
system=$2
cudaid1=$3
cudaid2=$4
mode_tag=$5
step=$6
# 这里怎么兼容cpu/gpu；linux、mac
if [[ ${mode_tag} == "CE" ]];then
    log_path=${log_path}/language_model_ernie/
    MAX_STEPS=100
    SAVE_STEPS=20
else
    log_path=$log_path
    MAX_STEPS=40
    SAVE_STEPS=20
fi
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
train(){
    cd ./language_model_ernie
    # 收敛性case
    if [[ ${mode_tag} == "CE_CON" ]];then
        # 收敛性的数据集准备TODO
        bash train.sh ${device} "multi" ${cudaid2} con > $log_path/train_multi_${device}.log 2>&1
        print_info $? train_multi_${device}
    else
        # 获取数据
        bash data_proc.sh
        # 单卡训练
        unset CUDA_VISIBLE_DEVICES
        bash train.sh ${device} "single" ${cudaid1} common ${MAX_STEPS} ${SAVE_STEPS} >$log_path/train_single_${device}.log 2>&1
        print_info $? train_single_${device}
        bash train.sh ${device} "multi" ${cudaid2} common ${MAX_STEPS} ${SAVE_STEPS} >$log_path/train_multi_${device}.log 2>&1
        print_info $? train_multi_${device}
    fi
    cd ..
}
if [[ ${step} && ${step} == "all" ]];then
    exec_list=(${default_list//,/ })
elif [[ ${step} ]];then
    exec_list=(${step//,/ })
else
    exec_list=(${default_list//,/ })
fi
echo ${exec_list[@]}
for case in ${exec_list[@]};do
    ${case}
done
}
# 10 xlnet
xlnet(){
# 固定接收5个参数，有扩展可以自定义
default_list="finetune"
device=$1
system=$2
cudaid1=$3
cudaid2=$4
mode_tag=$5
step=$6
# 这里怎么兼容cpu/gpu；linux、mac
if [[ ${mode_tag} == "CE" ]];then
    log_path=${log_path}/xlnet/
    MAX_STEPS=10
    SAVE_STEPS=10
    LOGGING_STEPS=1
else
    log_path=$log_path
    MAX_STEPS=1
    SAVE_STEPS=1
    LOGGING_STEPS=1
fi
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
finetune(){
    cd ./xlnet
    bash fine_tune.sh 'SST-2' ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/SST-2-fine_tune.log 2>&1
    print_info $? SST-2-fine_tune
    if [[ ${mode_tag} == "CE" ]];then
        bash fine_tune.sh 'CoLA' ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/CoLA-fine_tune.log 2>&1
        print_info $? CoLA-fine_tune
        bash fine_tune.sh 'MRPC' ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/MRPC-fine_tune.log 2>&1
        print_info $? MRPC-fine_tune
        bash fine_tune.sh 'STS-B' ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/STS-B-fine_tune.log 2>&1
        print_info $? STS-B-fine_tune
        bash fine_tune.sh 'QNLI' ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/QNLI-fine_tune.log 2>&1
        print_info $? QNLI-fine_tune
        bash fine_tune.sh 'QQP' ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/QQP-fine_tune.log 2>&1
        print_info $? QQP-fine_tune
        bash fine_tune.sh 'MNLI' ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/MNLI-fine_tune.log 2>&1
        print_info $? MNLI-fine_tune
        bash fine_tune.sh 'RTE' ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/RTE-fine_tune.log 2>&1
        print_info $? RTE-fine_tune
        bash fine_tune.sh 'WNLI' ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/WNLI-fine_tune.log 2>&1
        print_info $? WNLI-fine_tune
    fi
    cd ..
}
if [[ ${step} && ${step} == "all" ]];then
    exec_list=(${default_list//,/ })
elif [[ ${step} ]];then
    exec_list=(${step//,/ })
else
    exec_list=(${default_list//,/ })
fi
echo ${exec_list[@]}
for case in ${exec_list[@]};do
    ${case}
done
}
# 11 ofa
ofa(){
# 固定接收5个参数，有扩展可以自定义
default_list="finetune,train"
device=$1
system=$2
cudaid1=$3
cudaid2=$4
mode_tag=$5
step=$6
# 这里怎么兼容cpu/gpu；linux、mac
if [[ ${mode_tag} == "CE" ]];then
    log_path=${log_path}/ofa/
    MAX_STEPS=10
    SAVE_STEPS=10
    LOGGING_STEPS=1
    MODEL_STEP=10
else
    log_path=$log_path
    MAX_STEPS=1
    SAVE_STEPS=1
    LOGGING_STEPS=1
    MODEL_STEP=1
fi
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
finetune(){
    cd ./ofa
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash finetune.sh ${device} 'single' SST-2 ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_SST-2_single_${device}.log 2>&1
    print_info $? finetune_SST-2_single_${device}
    export CUDA_VISIBLE_DEVICES=${cudaid2}
    bash finetune.sh ${device} 'multi' SST-2  ${cudaid2} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_SST-2_multi_${device}.log 2>&1
    print_info $? finetune_SST-2_multi_${device}
    if [[ ${mode_tag} == "CE" ]];then
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash finetune.sh ${device} 'single' QNLI ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_QNLI_single_${device}.log 2>&1
        print_info $? finetune_QNLI_single_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid2}
        bash finetune.sh ${device} 'multi' QNLI ${cudaid2} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_QNLI_multi_${device}.log 2>&1
        print_info $? finetune_QNLI_multi_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash finetune.sh ${device} 'single' CoLA ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_CoLA_single_${device}.log 2>&1
        print_info $? finetune_CoLA_single_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid2}
        bash finetune.sh ${device} 'multi' CoLA ${cudaid2} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_CoLA_multi_${device}.log 2>&1
        print_info $? finetune_CoLA_multi_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash finetune.sh ${device} 'single' MRPC ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_MRPC_single_${device}.log 2>&1
        print_info $? finetune_MRPC_single_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid2}
        bash finetune.sh ${device} 'multi' MRPC ${cudaid2} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_MRPC_multi_${device}.log 2>&1
        print_info $? finetune_MRPC_multi_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash finetune.sh ${device} 'single' STS-B ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_STS-B_single_${device}.log 2>&1
        print_info $? finetune_STS-B_single_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid2}
        bash finetune.sh ${device} 'multi' STS-B ${cudaid2} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_STS-B_multi_${device}.log 2>&1
        print_info $? finetune_STS-B_multi_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash finetune.sh ${device} 'single' QQP ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_QQP_single_${device}.log 2>&1
        print_info $? finetune_QQP_single_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid2}
        bash finetune.sh ${device} 'multi' QQP ${cudaid2} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_QQP_multi_${device}.log 2>&1
        print_info $? finetune_QQP_multi_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash finetune.sh ${device} 'single' MNLI ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_MNLI_single_${device}.log 2>&1
        print_info $? finetune_MNLI_single_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid2}
        bash finetune.sh ${device} 'multi' MNLI ${cudaid2} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_MNLI_multi_${device}.log 2>&1
        print_info $? finetune_MNLI_multi_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash finetune.sh ${device} 'single' RTE ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_RTE_single_${device}.log 2>&1
        print_info $? finetune_RTE_single_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid2}
        bash finetune.sh ${device} 'multi' RTE ${cudaid2} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/finetune_RTE_multi_${device}.log 2>&1
        print_info $? finetune_RTE_multi_${device}
    fi
    cd ..
}
train(){
    cd ./ofa
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash train.sh ${device} 'single' SST-2 ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${MODEL_STEP} > $log_path/train_SST-2_single_${device}.log 2>&1
    print_info $? train_SST-2_single_${device}
    unset CUDA_VISIBLE_DEVICES
    bash train.sh ${device} 'multi' SST-2 ${cudaid2} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${MODEL_STEP} > $log_path/train_SST-2_multi_${device}.log 2>&1
    print_info $? train_SST-2_multi_${device}
    if [[ ${mode_tag} == "CE" ]];then
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash train.sh ${device} 'single' QNLI ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${MODEL_STEP} > $log_path/train_QNLI_single_${device}.log 2>&1
        print_info $? train_QNLI_single_${device}
        unset CUDA_VISIBLE_DEVICES
        bash train.sh ${device} 'multi' QNLI ${cudaid2} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${MODEL_STEP} > $log_path/train_QNLI_multi_${device}.log 2>&1
        print_info $? train_QNLI_multi_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash train.sh ${device} 'single' CoLA ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${MODEL_STEP} > $log_path/train_CoLA_single_${device}.log 2>&1
        print_info $? train_CoLA_single_${device}
        unset CUDA_VISIBLE_DEVICES
        bash train.sh ${device} 'multi' CoLA ${cudaid2} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${MODEL_STEP} > $log_path/train_CoLA_multi_${device}.log 2>&1
        print_info $? train_CoLA_multi_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash train.sh ${device} 'single' MRPC ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${MODEL_STEP} > $log_path/train_MRPC_single_${device}.log 2>&1
        print_info $? train_MRPC_single_${device}
        unset CUDA_VISIBLE_DEVICES
        bash train.sh ${device} 'multi' MRPC ${cudaid2} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${MODEL_STEP} > $log_path/train_MRPC_multi_${device}.log 2>&1
        print_info $? train_MRPC_multi_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash train.sh ${device} 'single' STS-B ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${MODEL_STEP} > $log_path/train_STS-B_single_${device}.log 2>&1
        print_info $? train_STS-B_single_${device}
        unset CUDA_VISIBLE_DEVICES
        bash train.sh ${device} 'multi' STS-B ${cudaid2} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${MODEL_STEP} > $log_path/train_STS-B_multi_${device}.log 2>&1
        print_info $? train_STS-B_multi_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash train.sh ${device} 'single' QQP ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${MODEL_STEP} > $log_path/train_QQP_single_${device}.log 2>&1
        print_info $? train_QQP_single_${device}
        unset CUDA_VISIBLE_DEVICES
        bash train.sh ${device} 'multi' QQP ${cudaid2} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${MODEL_STEP} > $log_path/train_QQP_multi_${device}.log 2>&1
        print_info $? train_QQP_multi_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash train.sh ${device} 'single' MNLI ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${MODEL_STEP} > $log_path/train_MNLI_single_${device}.log 2>&1
        print_info $? train_MNLI_single_${device}
        unset CUDA_VISIBLE_DEVICES
        bash train.sh ${device} 'multi' MNLI ${cudaid2} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${MODEL_STEP} > $log_path/train_MNLI_multi_${device}.log 2>&1
        print_info $? train_MNLI_multi_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash train.sh ${device} 'single' RTE ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${MODEL_STEP} > $log_path/train_RTE_single_${device}.log 2>&1
        print_info $? train_RTE_single_${device}
        unset CUDA_VISIBLE_DEVICES
        bash train.sh ${device} 'multi' RTE ${cudaid2} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} ${MODEL_STEP} > $log_path/train_RTE_multi_${device}.log 2>&1
        print_info $? train_RTE_multi_${device}
    fi
    cd ..
}
if [[ ${step} && ${step} == "all" ]];then
    exec_list=(${default_list//,/ })
elif [[ ${step} ]];then
    exec_list=(${step//,/ })
else
    exec_list=(${default_list//,/ })
fi
echo ${exec_list[@]}
for case in ${exec_list[@]};do
    ${case}
done
}
# 12 albert
albert(){
# 固定接收5个参数，有扩展可以自定义
default_list="train"
device=$1
system=$2
cudaid1=$3
cudaid2=$4
mode_tag=$5
step=$6
# 这里怎么兼容cpu/gpu；linux、mac
if [[ ${mode_tag} == "CE" ]];then
    log_path=${log_path}/glue/
    MAX_STEPS=10
    SAVE_STEPS=10
    LOGGING_STEPS=1
else
    log_path=$log_path
    MAX_STEPS=1
    SAVE_STEPS=1
    LOGGING_STEPS=1
fi
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
train(){
    cd ./glue
    # 单卡
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    bash train.sh ${device} 'single' 'albert' 'albert-base-v2' 'SST-2' 1e-5 ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/train_albert-base-v2_SST-2_single_${device}.log 2>&1
    print_info $? train_albert-base-v2_SST-2_single_${device}
    #多卡
    export CUDA_VISIBLE_DEVICES=${cudaid2}
    bash train.sh ${device} 'multi' 'albert' 'albert-base-v2' 'SST-2' 1e-5 ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/train_albert-base-v2_SST-2_multi_${device}.log 2>&1
    print_info $? train_albert-base-v2_SST-2_multi_${device}
    if [[ ${mode_tag} == "CE" ]];then
        # CE覆盖的多个数据集
        export CUDA_VISIBLE_DEVICES=${cudaid1}
        bash train.sh ${device} 'single' 'albert' 'albert-base-v2' 'MNLI' 3e-5 ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/train_albert-base-v2_MNLI_single_${device}.log 2>&1
        print_info $? train_albert-base-v2_MNLI_single_${device}
        export CUDA_VISIBLE_DEVICES=${cudaid2}
        bash train.sh ${device} 'multi' 'albert' 'albert-base-v2' 'MNLI' 3e-5 ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} > $log_path/train_albert-base-v2_MNLI_multi_${device}.log 2>&1
        print_info $? train_albert-base-v2_MNLI_multi_${device}
    fi
    cd ..
}
if [[ ${step} && ${step} == "all" ]];then
    exec_list=(${default_list//,/ })
elif [[ ${step} ]];then
    exec_list=(${step//,/ })
else
    exec_list=(${default_list//,/ })
fi
echo ${exec_list[@]}
for case in ${exec_list[@]};do
    ${case}
done
}
# 13 squad
squad(){
# 固定接收5个参数，有扩展可以自定义
default_list="train"
device=$1
system=$2
cudaid1=$3
cudaid2=$4
mode_tag=$5
step=$6
# 这里怎么兼容cpu/gpu；linux、mac
if [[ ${mode_tag} == "CE" ]];then
    log_path=${log_path}/SQuAD/
    MAX_STEPS=30
    SAVE_STEPS=10
    LOGGING_STEPS=10
    cd ./SQuAD
    bash data_proc.sh
    cd ..
else
    log_path=$log_path
    MAX_STEPS=1
    SAVE_STEPS=1
    LOGGING_STEPS=1
fi
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
train(){
    cd ./SQuAD
    bash train.sh ${device} 1.1 ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} >$log_path/train_${device}_1.1.log 2>&1
    print_info $? train_${device}_1.1
    if [[ ${mode_tag} == "CE" ]];then
        bash train.sh ${device} 2.0 ${cudaid1} ${MAX_STEPS} ${SAVE_STEPS} ${LOGGING_STEPS} >$log_path/train_${device}_2.0.log 2>&1
        print_info $? train_${device}_2.0
    fi
    cd ..
}
if [[ ${step} && ${step} == "all" ]];then
    exec_list=(${default_list//,/ })
elif [[ ${step} ]];then
    exec_list=(${step//,/ })
else
    exec_list=(${default_list//,/ })
fi
echo ${exec_list[@]}
for case in ${exec_list[@]};do
    ${case}
done
}


####################################
# 程序的调度入口
model_func=$1
args=""
if [ $# -gt 0 ]; then
    args=${@:2}
fi
echo $args
echo $model_func
${model_func} $args
