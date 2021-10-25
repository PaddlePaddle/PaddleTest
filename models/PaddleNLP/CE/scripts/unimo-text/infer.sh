#unset http_proxy
HTTPPROXY=$http_proxy
HTTPSPROXY=$https_proxy
unset http_proxy
unset https_proxy

#外部传入参数说明
# $1:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/text_generation/unimo-text/

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

if [ $2 == 'faster' ];then
    #  先编包
    cd ../../../paddlenlp/ops
    mkdir build
    cd build/
    cmake .. -DSM=70 -DCMAKE_BUILD_TYPE=Release -DPY_CMD=python3.8 -DWITH_UNIFIED=ON
    make -j
    cd $code_path/faster_unimo
    python infer.py \
        --dataset_name=dureader_qg \
        --decoding_lib ../../../../paddlenlp/ops/build/lib/libdecoding_op.so \
        --model_name_or_path=../unimo/checkpoints/single/model_908/ \
        --logging_steps=100 \
        --batch_size=16 \
        --max_seq_len=512 \
        --max_target_len=30 \
        --max_dec_len=20 \
        --min_dec_len=3 \
        --decode_strategy=sampling \
        --device=$1 \
        --top_k 0 \
        --top_p 1 >$log_path/infer_$2_$1.log 2>&1
    print_info $? infer_$2_$1
else
    python run_gen.py \
        --dataset_name=dureader_qg \
        --model_name_or_path=./unimo/checkpoints/single/model_908/ \
        --logging_steps=100 \
        --batch_size=16 \
        --max_seq_len=512 \
        --max_target_len=30 \
        --do_predict \
        --max_dec_len=20 \
        --min_dec_len=3 \
        --device=$1 >$log_path/infer_$2_$1.log 2>&1
    print_info $? infer_$2_$1
fi



#set http_proxy
export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
