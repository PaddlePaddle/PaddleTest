#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'recv' 恢复训练
# $2:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型python预测部署阶段"


code_path=${nlp_dir}/examples/information_extraction/waybill_ie/

cd $code_path

python export_model.py --params_path $2_ckpt/model_80/model_state.pdparams --output_path=./$2_output

python deploy/python/predict.py --model_dir ./$2_output
