
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"
#配置目标数据存储路径
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/model_compression/distill_lstm/
data_path=$code_path/SST-2/
cd $code_path
#获取数据逻辑
#数据处理逻辑
if [ ! -f "senta_word_dict.txt" ]
then
    wget https://paddlenlp.bj.bcebos.com/data/senta_word_dict.txt
fi

rm -rf $data_path
mkdir -p $data_path
cp -r /workspace/task/datasets/distill_lstm/sst-2/*  ./SST-2/
