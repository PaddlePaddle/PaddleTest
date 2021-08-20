

#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'recv' 恢复训练
# $2:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/text_summarization/pointer_summarizer
log_path=$root_path/log/$model_name/
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi

#访问RD程序
cd $code_path

sed -i 's/max_iterations = 100000/max_iterations = 30/g' config.py
sed -i 's/if iter % 5000 == 0 or iter == 1000:/if iter % 30 == 0 :/g' train.py

#训练
python train.py > $log_path/train_$2_$1.log 2>&1
