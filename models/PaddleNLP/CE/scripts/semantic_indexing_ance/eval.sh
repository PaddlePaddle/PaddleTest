#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'recv' 恢复训练
# $2:  $XPU = gpu or cpu
#获取当前路径
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
code_path=$cur_path/../../models_repo/examples/semantic_indexing/
log_path=$root_path/log/$model_name/
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi

#删除分布式日志重新记录
rm -rf $code_path/log/workerlog.0



print_info(){
if [ $1 -ne 0 ];then
    echo "exit_code: 1.0" >> ${log_path}/$2.log
else
    echo "exit_code: 0.0" >> ${log_path}/$2.log
fi
}

#访问RD程序
cd $code_path

mkdir -p recall_batch_neg
mkdir -p recall_hardest_neg

if [[ $3 == "batch" ]]; then
    python -u -m paddle.distributed.launch --gpus "$2" --log_dir "recall_log/" \
        recall.py \
        --device $1 \
        --recall_result_dir "recall_batch_neg" \
        --recall_result_file "recall_result.txt" \
        --params_path "checkpoints_batch_neg_single/model_1000/model_state.pdparams" \
        --hnsw_m 100 \
        --hnsw_ef 100 \
        --batch_size 128 \
        --output_emb_size 256\
        --max_seq_length 60 \
        --recall_num 50 \
        --similar_text_pair "same_semantic.tsv" \
        --corpus_file "corpus_file"

    python -u evaluate.py \
        --similar_text_pair "same_semantic.tsv" \
        --recall_result_file "./recall_batch_neg/recall_result.txt" \
        --recall_num 50 > ${log_path}/eval_$3_$1.log 2>&1
    print_info $?  eval_$3_$1
else
    # 先支持GPU
    python -u -m paddle.distributed.launch --gpus "$2" --log_dir "recall_log/" \
        recall.py \
        --device $1 \
        --recall_result_dir "recall_hardest_neg" \
        --recall_result_file "recall_result.txt" \
        --params_path "checkpoints_hardest_neg_single/model_1000/model_state.pdparams" \
        --hnsw_m 100 \
        --hnsw_ef 100 \
        --batch_size 128 \
        --output_emb_size 256\
        --max_seq_length 60 \
        --recall_num 50 \
        --similar_text_pair "same_semantic.tsv" \
        --corpus_file "corpus_file"

    python -u evaluate.py \
        --similar_text_pair "same_semantic.tsv" \
        --recall_result_file "./recall_hardest_neg/recall_result.txt" \
        --recall_num 50 > ${log_path}/eval_$3_$1.log 2>&1
    print_info $?  eval_$3_$1
fi

export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
