model_name=${1:-"xlnet"}
cards=${2:-"1"}
if [ $cards = "1" ];then
     export CUDA_VISIBLE_DEVICES=0
elif [ $cards = "8" ];then
     export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
else
    echo -n "cards error"
    exit -1
fi
if [ $model_name = "xlnet" ];then
    CUR_PATH=`pwd`
    git clone https://github.com/PaddlePaddle/PaddleNLP.git
    rm PaddleNLP/tests/benchmark/xlnet/run_all.sh
    rm PaddleNLP/tests/benchmark/xlnet/run_benchmark.sh
    cp run_all.sh PaddleNLP/tests/benchmark/xlnet
    cp run_benchmark.sh PaddleNLP/tests/benchmark/xlnet
    cd PaddleNLP
    bash tests/benchmark/xlnet/run_all.sh $cards $CUR_PATH >log.1 2>&1
    final_res=`tail -1 log.xlnet | awk '{print $NF}' 2>/dev/null`
    if [ -z ${final_res} ];then
        echo -n "trian error"
        exit 1
    fi
    cd ..
    python is_conv.py ${final_res} 0.9 >log.conv 2>&1
    flag=$?
    if [ $flag == 1 ]; then
        echo -n "trian error"
        exit 1
    elif [ $flag == 2 ];then
        echo -n "not convergent"
        exit 2
    fi
    cat log.conv
else
    echo -n "model_name error"
    exit 3
fi
