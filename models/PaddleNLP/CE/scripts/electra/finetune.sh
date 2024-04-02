cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型Fine-tune阶段"

code_path=${nlp_dir}/model_zoo/$model_name/

#访问RD程序
cd $code_path
python -u ./run_glue.py \
    --model_type electra \
    --model_name_or_path electra-small \
    --task_name SST-2 \
    --max_seq_length 128 \
    --batch_size 32   \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --save_steps 10 \
    --output_dir ./SST-2/ \
    --max_steps 50\
    --device gpu
