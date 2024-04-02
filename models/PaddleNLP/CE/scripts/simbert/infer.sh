#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'recv' 恢复训练
# $2:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型预测阶段"

#路径配置
code_path=${nlp_dir}/examples/text_matching/simbert/


#删除分布式日志重新记录
rm -rf $code_path/log/workerlog.0


#访问RD程序
cd $code_path

python predict.py --input_file ./dev.tsv  --device $1
