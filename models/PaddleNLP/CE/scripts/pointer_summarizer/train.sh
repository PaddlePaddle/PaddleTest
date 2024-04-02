

#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'recv' 恢复训练
# $2:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"

#路径配置
code_path=${nlp_dir}/examples/text_summarization/pointer_summarizer

MAX_STEPS=$3
#访问RD程序
cd $code_path

sed -i "s/max_iterations = 100000/max_iterations = ${MAX_STEPS}/g" config.py
sed -i "s/if iter % 5000 == 0 or iter == 1000:/if iter % ${MAX_STEPS} == 0 :/g" train.py

#训练
python train.py
