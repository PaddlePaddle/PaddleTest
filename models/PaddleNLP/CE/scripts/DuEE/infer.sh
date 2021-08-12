

cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"
#取消代理
HTTPPROXY=$http_proxy
HTTPSPROXY=$https_proxy
unset http_proxy
unset https_proxy

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/information_extraction/DuEE/
log_path=$root_path/log/$model_name/

if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi



print_info(){
if [ $1 -ne 0 ];then
    cat ${log_path}/$2.log
    echo "exit_code: 1.0" >> ${log_path}/$2.log
else
    echo "exit_code: 0.0" >> ${log_path}/$2.log
fi
}

#访问RD程序
cd $code_path

# data_dir=./data/DuEE-Fin
# conf_path=./conf/DuEE-Fin
# ckpt_dir=./ckpt/DuEE-Fin
# predict_data=./data/DuEE-Fin/sentence/test.json

echo -e "check and create directory"
dir_list=(./ckpt ${ckpt_dir} ./submit)
for item in ${dir_list[*]}
do
    if [ ! -d ${item} ]; then
        mkdir ${item}
        echo "create dir * ${item} *"
    else
        echo "dir ${item} exist"
    fi
done

if [[ $2 == "enum" ]]; then
    python classifier.py \
        --num_epoch 1 \
        --learning_rate 5e-5 \
        --tag_path ./conf/DuEE-Fin/$2_tag.dict \
        --train_data ./data/DuEE-Fin/$2/train.tsv \
        --dev_data ./data/DuEE-Fin/$2/dev.tsv \
        --test_data ./data/DuEE-Fin/$2/test.tsv \
        --predict_data ./data/DuEE-Fin/sentence/test.json \
        --do_train False \
        --do_predict True \
        --max_seq_len 300 \
        --batch_size 16 \
        --skip_step 1 \
        --valid_step 1 \
        --checkpoints ./ckpt/DuEE-Fin/$2 \
        --init_ckpt ./ckpt/DuEE-Fin/$2/final.pdparams \
        --predict_save_path ./ckpt/DuEE-Fin/$2/test_pred.json \
        --device $1 > $log_path/infer_$2_$3_$1.log 2>&1
    print_info $? infer_$2_$3_$1
else
    python sequence_labeling.py \
            --num_epoch 1\
            --learning_rate 5e-5 \
            --tag_path ./conf/DuEE-Fin/$2_tag.dict \
            --train_data ./data/DuEE-Fin/$2/train.tsv \
            --dev_data ./data/DuEE-Fin/$2/dev.tsv \
            --test_data ./data/DuEE-Fin/$2/test.tsv \
            --predict_data ./data/DuEE-Fin/sentence/test.json \
            --do_train False \
            --do_predict True \
            --max_seq_len 300 \
            --batch_size 1 \
            --skip_step 10 \
            --valid_step 50 \
            --checkpoints ./ckpt/DuEE-Fin/$2 \
            --init_ckpt ./ckpt/DuEE-Fin/$2/best.pdparams \
            --predict_save_path ./ckpt/DuEE-Fin/$2/test_pred.json \
            --device $1 > $log_path/infer_$2_$3_$1.log 2>&1
    print_info $? infer_$2_$3_$1
fi
