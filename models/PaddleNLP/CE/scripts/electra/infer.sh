cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型样例测试阶段"

code_path=${nlp_dir}/model_zoo/$model_name/


cd $code_path
python -u ./export_model.py \
    --input_model_dir ./SST-2/sst-2_ft_model_40.pdparams/ \
    --output_model_dir ./ \
    --model_name electra-deploy


python -u ./deploy/python/predict.py \
    --model_file ./electra-deploy.pdmodel \
    --params_file ./electra-deploy.pdiparams \
    --predict_sentences "uneasy mishmash of styles and genres ." "director rob marshall went out gunning to make a great one ." \
    --batch_size 2 \
    --max_seq_length 128 \
    --model_name electra-small
