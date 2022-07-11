#外部传入参数说明
# $1:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型预测阶段"

#路径配置
code_path=${nlp_dir}/examples/text_matching/question_matching

MODEL_STEP=$2

#访问RD程序
cd $code_path

python -u predict.py \
    --device $1 \
    --params_path "./checkpoints/single/model_${MODEL_STEP}/model_state.pdparams" \
    --batch_size 128 \
    --input_file ./data/test/public_test_A \
    --result_file "predict_result"
