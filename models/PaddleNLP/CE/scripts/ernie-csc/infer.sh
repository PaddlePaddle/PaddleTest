#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"
#配置目标数据存储路径
code_path=${nlp_dir}/examples/text_correction/ernie-csc/


cd $code_path

python predict_sighan.py \
    --model_name_or_path ernie-1.0 \
    --test_file sighan_test/sighan13/input.txt \
    --batch_size 32 \
    --ckpt_path checkpoints/single/best_model.pdparams \
    --predict_file predict_sighan13.txt

