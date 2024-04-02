cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型Fine-tune阶段"

#路径配置
code_path=${nlp_dir}/examples/language_model/$model_name/

cd $code_path

sed -i "s#init_from_checkpoint: .*#init_from_checkpoint: \"./trained_models/step_5\"#g" $code_path/configs/enwik8.yaml
sed -i "s#init_from_params: .*#init_from_params: \"./trained_models/step_5\"#g" $code_path/configs/enwik8.yaml
python ./eval.py --config ./configs/enwik8.yaml
