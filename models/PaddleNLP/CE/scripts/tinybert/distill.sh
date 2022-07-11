cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型蒸馏阶段"

NAME=$(echo $3 | tr 'A-Z' 'a-z')

#路径配置
code_path=${nlp_dir}/model_zoo/tinybert/
data_path=${nlp_dir}/examples/benchmark/glue/tmp/$3/$2/${NAME}_ft_model_30.pdparams
student_model_name=tinybert-6l-768d-v2

MAX_STEPS=$4
SAVE_STEPS=$5
LOGGING_STEPS=$6
TEACHER_PATH=$7
STUDENT_NAME=$8
if [[ ${TEACHER_PATH} ]];then
    data_path=${TEACHER_PATH}
fi
if [[ ${STUDENT_NAME} ]];then
    student_model_name=${STUDENT_NAME}
fi
#访问RD程序
cd $code_path

python task_distill.py \
    --model_type tinybert \
    --student_model_name_or_path ${student_model_name} \
    --task_name $3 \
    --intermediate_distill \
    --max_seq_length 64 \
    --batch_size 32   \
    --T 1 \
    --teacher_model_type bert \
    --teacher_path $data_path \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --max_steps ${MAX_STEPS} \
    --output_dir ./tmp/$3/$2 \
    --device $1
