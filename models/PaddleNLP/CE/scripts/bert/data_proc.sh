
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"

#配置目标数据存储路径
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/language_model/$model_name

#数据处理逻辑
cd $code_path

python create_pretraining_data.py \
  --input_file=data/sample_text.txt \
  --output_file=data/training_data.hdf5 \
  --bert_model=bert-base-uncased \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
