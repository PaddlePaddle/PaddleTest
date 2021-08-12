cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型Fine-tune阶段"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/language_model/$model_name/
log_path=$root_path/log/$model_name/

if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
cd $code_path
python run_classifier.py --model_name_or_path $2 \
    --output_dir "output_eval" \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --max_steps 10 \
    --save_steps 10 \
    --max_encoder_length 3072 > $log_path/$2-eval.log 2>&1

cat $log_path/$2-eval.log
