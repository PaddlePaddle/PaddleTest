#!/usr/bin/env bash

print_info(){
if [ $1 -ne 0 ];then
    mv ${log_path}/$2 ${log_path}/$2_FAIL.log
    echo -e "\033[31m ${log_path}/$2_FAIL \033[0m"
else
    mv ${log_path}/$2 ${log_path}/$2_SUCCESS.log
    echo -e "\033[32m ${log_path}/$2_SUCCESS \033[0m"
fi
}
# case list
# 1 waybill_ie (无可控参数，数据集外置)
waybill_ie(){
cd ${nlp_dir}/examples/information_extraction/waybill_ie/
cp -r /ssd1/paddlenlp/download/waybill_ie/* ${nlp_dir}/examples/information_extraction/waybill_ie/data/
export CUDA_VISIBLE_DEVICES=${cudaid1}
# BiGRU +CRF star training
time (
python download.py --data_dir ./waybill_ie
python run_bigru_crf.py >${log_path}/waybill_ie_bigru_crf) >>${log_path}/waybill_ie_bigru_crf 2>&1
print_info $? waybill_ie_bigru_crf
# ERNIE +RF star training
time (python run_ernie.py >${log_path}/waybill_ie_ernie) >>${log_path}/waybill_ie_ernie 2>&1
print_info $? waybill_ie_ernie
# ERNIE +CRF star training
time (python run_ernie_crf.py >${log_path}/waybill_ie_ernie_crf) >>${log_path}/waybill_ie_ernie_crf 2>&1
print_info $? waybill_ie_ernie_crf
}

# 2 msra_ner （不可控，内置）
msra_ner(){
cd ${nlp_dir}/examples/information_extraction/msra_ner/
CUDA_VISIBLE_DEVICES=${cudaid2}
## train
time (python -m paddle.distributed.launch  ./train.py \
    --model_name_or_path bert-base-multilingual-uncased \
    --max_seq_length 128 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --save_steps 700 \
    --output_dir ./tmp/msra_ner/ \
    --device gpu >${log_path}/msra_ner_train) >>${log_path}/msra_ner_train 2>&1
print_info $? msra_ner_train
## eval
time (python -u ./eval.py \
    --model_name_or_path bert-base-multilingual-uncased \
    --max_seq_length 128 \
    --batch_size 16 \
    --device gpu \
    --init_checkpoint_path tmp/msra_ner/model_700.pdparams >${log_path}/msra_ner_eval) >>${log_path}/msra_ner_eval 2>&1
print_info $? msra_ner_eval
## predict
time (python -u ./predict.py \
    --model_name_or_path bert-base-multilingual-uncased \
    --max_seq_length 128 \
    --batch_size 16 \
    --device gpu \
    --init_checkpoint_path tmp/msra_ner/model_700.pdparams >${log_path}/msra_ner_predict) >>${log_path}/msra_ner_predict 2>&1
print_info $? msra_ner_predict
}
# 3 glue
glue() {
cd ${nlp_dir}/examples/benchmark/glue/
CUDA_VISIBLE_DEVICES=${cudaid2}
##  TASK_SST-2
export TASK_NAME=SST-2
time (python -m paddle.distributed.launch  run_glue.py \
    --model_type albert    \
    --model_name_or_path albert-base-v2    \
    --task_name $TASK_NAME \
    --max_seq_length 128   \
    --batch_size 32    \
    --learning_rate 1e-5    \
    --max_steps 1    \
    --warmup_steps 1256    \
    --logging_steps 1    \
    --save_steps 1   \
    --output_dir ./tmp/$TASK_NAME/    \
    --device gpu  >${log_path}/glue_${TASK_NAME}_train) >>${log_path}/glue_${TASK_NAME}_train 2>&1
print_info $? glue_${TASK_NAME}_train
}
# 4 bert
bert() {
CUDA_VISIBLE_DEVICES=${cudaid2}
cd ${nlp_dir}/examples/language_model/bert/
cp -r /ssd1/paddlenlp/download/bert/* ./data/
## pretrain
time (python -m paddle.distributed.launch run_pretrain.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --max_predictions_per_seq 20 \
    --batch_size 32  \
    --learning_rate 1e-4 \
    --weight_decay 1e-2 \
    --adam_epsilon 1e-6 \
    --warmup_steps 10000 \
    --input_dir data/hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/wikicorpus_en/training/ \
    --output_dir pretrained_models/ \
    --logging_steps 1 \
    --save_steps 1 \
    --max_steps 1 \
    --device gpu \
    --use_amp False >${log_path}/bert_pretrain) >>${log_path}/bert_pretrain 2>&1
print_info $? bert_pretrain
time (python -m paddle.distributed.launch run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name SST-2 \
    --max_seq_length 128 \
    --batch_size 32   \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --save_steps 1 \
    --max_steps 1 \
    --output_dir ./tmp/ \
    --device gpu \
    --use_amp False >${log_path}/bert_fintune) >>${log_path}/bert_fintune 2>&1
print_info $? bert_fintune
time (python -u ./export_model.py \
    --model_type bert \
    --model_path bert-base-uncased \
    --output_path ./infer_model/model >${log_path}/bert_export) >>${log_path}/bert_export 2>&1
print_info $? bert_export
time (python -u ./predict_glue.py \
    --task_name SST-2 \
    --model_type bert \
    --model_path ./infer_model/model \
    --batch_size 32 \
    --max_seq_length 128 >${log_path}/bert_predict) >>${log_path}/bert_predict 2>&1
print_info $? bert_predict
 }
# 5 skep (max save 不可控 内置)
skep () {
cd ${nlp_dir}/examples/sentiment_analysis/skep/
CUDA_VISIBLE_DEVICES=${cudaid2}
## train_sentence
time ( python -m paddle.distributed.launch train_sentence.py --batch_size 16 --epochs 1 --model_name "skep_ernie_1.0_large_ch" --device gpu --save_dir ./checkpoints >${log_path}/train_sentence) >>${log_path}/train_sentence 2>&1
print_info $? train_sentence
## train_aspect
time ( python -m paddle.distributed.launch train_aspect.py --batch_size 4 --epochs 1  --device gpu --save_dir ./aspect_checkpoints  >${log_path}/train_aspect) >>${log_path}/train_aspect 2>&1
print_info $? train_aspect
# # train_opinion
time ( python -m paddle.distributed.launch train_opinion.py  --batch_size 4 --epochs 1 --device gpu --save_dir ./opinion_checkpoints >${log_path}/train_opinion) >>${log_path}/train_opinion 2>&1
print_info $? train_opinion
# predict_sentence
time (python predict_sentence.py --model_name "skep_ernie_1.0_large_ch"  --params_path checkpoints/model_100/model_state.pdparams >${log_path}/predict_sentence) >>${log_path}/predict_sentence 2>&1
print_info $? predict_sentence
## predict_aspect
time (python predict_aspect.py --device 'gpu' --params_path ./aspect_checkpoint/model_100/model_state.pdparams  >${log_path}/predict_aspect) >>${log_path}/predict_aspect 2>&1
print_info $? predict_aspect
# # predict_opinion
time (python predict_opinion.py --device 'gpu' --params_path ./opinion_checkpoints/model_100/model_state.pdparams >${log_path}/predict_opinion) >>${log_path}/predict_opinion 2>&1
print_info $? predict_opinion
}
# 6 bigbird
bigbird(){
cd ${nlp_dir}/examples/language_model/bigbird/
CUDA_VISIBLE_DEVICES=${cudaid2}
time (python -m paddle.distributed.launch  --log_dir log  run_pretrain.py --model_name_or_path bigbird-base-uncased \
    --input_dir "./data" \
    --output_dir "output" \
    --batch_size 4 \
    --weight_decay 0.01 \
    --learning_rate 1e-5 \
    --max_steps 1 \
    --save_steps 1 \
    --logging_steps 1 \
    --max_encoder_length 512 \
    --max_pred_length 75 >${log_path}/bigbird_pretrain) >>${log_path}/bigbird_pretrain 2>&1
    print_info $? bigbird_pretrain
}
# 7 electra
electra(){
cd ${nlp_dir}/examples/language_model/electra/
CUDA_VISIBLE_DEVICES=${cudaid2}
export DATA_DIR=./BookCorpus/
cp -r /ssd1/paddlenlp/download/electra/BookCorpus/ ./
time (python -u ./run_pretrain.py \
    --model_type electra \
    --model_name_or_path electra-small \
    --input_dir ./BookCorpus/ \
    --output_dir ./pretrain_model/ \
    --train_batch_size 64 \
    --learning_rate 5e-4 \
    --max_seq_length 128 \
    --weight_decay 1e-2 \
    --adam_epsilon 1e-6 \
    --warmup_steps 10000 \
    --num_train_epochs 4 \
    --logging_steps 1 \
    --save_steps 1 \
    --max_steps 1 \
    --device gpu >${log_path}/electra_pretrain) >>${log_path}/electra_pretrain 2>&1
print_info $? electra_pretrain
}
# 8 gpt
gpt(){
cd ${nlp_dir}/examples/language_model/gpt/
cp -r /ssd1/paddlenlp/download/gpt/data/ ./
cp -r /ssd1/paddlenlp/download/gpt/ckpt/ ./
CUDA_VISIBLE_DEVICES=${cudaid2}
#pretrain
time (python -m paddle.distributed.launch run_pretrain.py \
    --model_type gpt \
    --model_name_or_path gpt2-en \
    --input_dir "./data" \
    --output_dir "output" \
    --weight_decay 0.01 \
    --grad_clip 1.0 \
    --max_steps 1 \
    --save_steps 1 \
    --decay_steps 1 \
    --warmup_rate 0.01 \
    --micro_batch_size 4 \
    --device gpu >${log_path}/gpt_pretrain) >>${log_path}/gpt_pretrain 2>&1
print_info $? gpt_pretrain
# test acc
cd tests/
time (python -m unittest test_accuracy.py >${log_path}/gpt_test_acc) >>${log_path}/gpt_test_acc 2>&1
print_info $? gpt_test_acc
}
# 9 xlnet
xlnet(){
cd ${nlp_dir}/examples/language_model/xlnet/
CUDA_VISIBLE_DEVICES=${cudaid2}
time (python -m paddle.distributed.launch ./run_glue.py \
    --model_name_or_path xlnet-base-cased \
    --task_name SST-2 \
    --max_seq_length 128 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --max_steps 1 \
    --logging_steps 1 \
    --save_steps 1 \
    --output_dir ./xlnet/ >${log_path}/xlnet_train) >>${log_path}/xlnet_train 2>&1
print_info $? xlnet_train
}
# 10 ofa
ofa(){
cd ${nlp_dir}/examples/model_compression/ofa/
cd ../../benchmark/glue/
CUDA_VISIBLE_DEVICES=${cudaid2}
# finetuing
time (python -u ./run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name SST-2 \
    --max_seq_length 128 \
    --batch_size 32   \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --max_steps 1 \
    --logging_steps 1 \
    --save_steps 1 \
    --output_dir ./ \
    --device gpu  >${log_path}/ofa_pretrain) >>${log_path}/ofa_pretrain 2>&1
print_info $? ofa_pretrain
mv sst-2_ft_model_1.pdparams/  ${nlp_dir}/examples/model_compression/ofa/
cd -
#model slim
CUDA_VISIBLE_DEVICES=${cudaid2}
time (python -m paddle.distributed.launch run_glue_ofa.py  \
          --model_type bert \
          --model_name_or_path ./sst-2_ft_model_1.pdparams/ \
          --task_name SST-2 --max_seq_length 128     \
          --batch_size 32       \
          --learning_rate 2e-5     \
          --num_train_epochs 1     \
          --max_steps 1 \
          --logging_steps 1    \
          --save_steps 1     \
          --output_dir ./ofa/SST-2 \
          --device gpu  \
          --width_mult_list 1.0 0.8333333333333334 0.6666666666666666 0.5 >${log_path}/ofa_slim) >>${log_path}/ofa_slim 2>&1
print_info $? ofa_slim
}
# 11 albert
albert (){
cd ${nlp_dir}/examples/benchmark/glue/
CUDA_VISIBLE_DEVICES=${cudaid2}
time (python -m paddle.distributed.launch  run_glue.py \
        --model_type albert    \
        --model_name_or_path albert-base-v2    \
        --task_name SST-2 \
        --max_seq_length 128   \
        --batch_size 32    \
        --learning_rate 1e-5    \
        --max_steps 1    \
        --warmup_steps 1256    \
        --logging_steps 1    \
        --save_steps 1   \
        --output_dir ./albert/SST-2/    \
        --device gpu >${log_path}/albert_sst-2_train) >>${log_path}/albert_sst-2_train 2>&1
print_info $? albert_sst-2_train
}
# #12 ernie 超过1h
# ernie (){
# CUDA_VISIBLE_DEVICES=${cudaid2}
# cd ${nlp_dir}/examples/language_model/ernie/
# time (python -m paddle.distributed.fleet.launch \
#     --log_dir ./output_dir/log \
#     run_pretraining.py \
#     --global_bsz 64 \
#     --micro_bsz 1 \
#     --max_seq_len 512 \
#     --ernie_config_file config/ernie_base_config.json \
#     --learning_rate 1e-4 \
#     --log_steps 1 \
#     --num_train_steps 1 \
#     --save_steps 100 \
#     --output_dir ./output_dir \
#     --use_recompute true \
#     --use_sharding true \
#     --use_sop false \
#     --num_mp=1 \
#     --num_sharding=2 \
#     --num_pp=1 \
#     --num_dp=1 >${log_path}/ernie_pretrain) >>${log_path}/ernie_pretrain 2>&1
# print_info $? ernie_pretrain
# }
# 13 squad
squad (){
cd ${nlp_dir}/examples/machine_reading_comprehension/SQuAD/
CUDA_VISIBLE_DEVICES=${cudaid1}
# finetune
time (python -m paddle.distributed.launch run_squad.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --max_seq_length 384 \
    --batch_size 12 \
    --learning_rate 3e-5 \
    --num_train_epochs 1 \
    --max_steps 1 \
    --logging_steps 1 \
    --save_steps 1 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --output_dir ./tmp/squad/ \
    --device gpu \
    --do_train \
    --do_predict >${log_path}/squad_train) >>${log_path}/squad_train 2>&1
print_info $? squad_train
# export model
time (python  -u ./export_model.py \
    --model_type bert \
    --model_path ./tmp/squad/model_1/ \
    --output_path ./infer_model/model >${log_path}/squad_export) >>${log_path}/squad_export 2>&1
print_info $? squad_export
# predict
time (python -u deploy/python/predict.py \
    --model_type bert \
    --model_name_or_path ./infer_model/model \
    --batch_size 2 \
    --max_seq_length 384 >${log_path}/squad_predict) >>${log_path}/squad_predict 2>&1
print_info $? squad_predict
}
# 14 tinybert
tinybert() {
CUDA_VISIBLE_DEVICES=${cudaid1}
cd ${nlp_dir}/examples/model_compression/tinybert/
cp -r /ssd1/paddlenlp/download/tinybert/pretrained_models/ ./
#中间层蒸馏
time (python task_distill.py \
    --model_type tinybert \
    --student_model_name_or_path tinybert-6l-768d-v2 \
    --task_name SST-2 \
    --intermediate_distill \
    --max_seq_length 64 \
    --batch_size 32   \
    --T 1 \
    --teacher_model_type bert \
    --teacher_path ./pretrained_models/SST-2/best_model_610/ \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --max_steps 1 \
    --logging_steps 1 \
    --save_steps 1 \
    --output_dir ./mid/SST-2/ \
    --device gpu >${log_path}/tinybert_midslim) >>${log_path}/tinybert_midslim 2>&1
print_info $? tinybert_midslim
#预测层蒸馏
time (python task_distill.py \
    --model_type tinybert \
    --student_model_name_or_path ./mid/SST-2/intermediate_distill_model_final.pdparams \
    --task_name SST-2 \
    --max_seq_length 64 \
    --batch_size 32   \
    --T 1 \
    --teacher_model_type bert \
    --teacher_path ./pretrained_models/SST-2/best_model_610/  \
    --learning_rate 3e-5 \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --max_steps 1 \
    --save_steps 1 \
    --output_dir ./ped/SST-2/ \
    --device gpu >${log_path}/tinybert_predslim) >>${log_path}/tinybert_predslim 2>&1
print_info $? tinybert_predslim
}
# 15 lexical_analysis
lexical_analysis(){
CUDA_VISIBLE_DEVICES=${cudaid2}
cd ${nlp_dir}/examples/lexical_analysis/
#train
time (python download.py --data_dir ./ )
time (python -m paddle.distributed.launch train.py \
        --data_dir ./lexical_analysis_dataset_tiny \
        --model_save_dir ./save_dir \
        --epochs 1 \
        --save_steps 15 \
        --logging_steps 1\
        --batch_size 32 \
        --device gpu >${log_path}/lexical_analysis_train) >>${log_path}/lexical_analysis_train 2>&1
print_info $? lexical_analysis_train
#export
time (python export_model.py \
    --data_dir=./lexical_analysis_dataset_tiny \
    --params_path=./save_dir/model_15.pdparams \
    --output_path=./infer_model/static_graph_params >${log_path}/lexical_analysis_export) >>${log_path}/lexical_analysis_export 2>&1
print_info $? lexical_analysis_export
# predict
time (python predict.py --data_dir ./lexical_analysis_dataset_tiny \
        --init_checkpoint ./save_dir/model_15.pdparams \
        --batch_size 32 \
        --device gpu >${log_path}/lexical_analysis_predict) >>${log_path}/lexical_analysis_predict 2>&1
print_info $? lexical_analysis_predict
# deploy
time (python deploy/predict.py \
    --model_file=infer_model/static_graph_params.pdmodel \
    --params_file=infer_model/static_graph_params.pdiparams \
    --data_dir lexical_analysis_dataset_tiny >${log_path}/lexical_analysis_deploy) >>${log_path}/lexical_analysis_deploy 2>&1
print_info $? lexical_analysis_deploy
}
# 16 seq2seq
seq2seq() {
CUDA_VISIBLE_DEVICES=${cudaid2}
cd ${nlp_dir}/examples/machine_translation/seq2seq/
# train  (1041/steps) 5min
time (python train.py \
    --num_layers 2 \
    --hidden_size 512 \
    --batch_size 128 \
    --max_epoch 1 \
    --log_freq 1 \
    --dropout 0.2 \
    --init_scale  0.1 \
    --max_grad_norm 5.0 \
    --device gpu \
    --model_path ./attention_models >${log_path}/seq2seq_train) >>${log_path}/seq2seq_train 2>&1
print_info $? seq2seq_train
# predict
time (python predict.py \
     --num_layers 2 \
     --hidden_size 512 \
     --batch_size 128 \
     --dropout 0.2 \
     --init_scale  0.1 \
     --max_grad_norm 5.0 \
     --init_from_ckpt attention_models/0 \
     --infer_output_file infer_output.txt \
     --beam_size 10 \
     --device gpu  >${log_path}/seq2seq_predict) >>${log_path}/seq2seq_predict 2>&1
print_info $? seq2seq_predict
# export
time (python export_model.py \
     --num_layers 2 \
     --hidden_size 512 \
     --batch_size 128 \
     --dropout 0.2 \
     --init_scale  0.1 \
     --max_grad_norm 5.0 \
     --init_from_ckpt attention_models/0.pdparams \
     --beam_size 10 \
     --export_path ./infer_model/model >${log_path}/seq2seq_export) >>${log_path}/seq2seq_export 2>&1
print_info $? seq2seq_export
# depoly
time (cd deploy/python
python infer.py \
    --export_path ../../infer_model/model \
    --device gpu \
    --batch_size 128 \
    --infer_output_file infer_output.txt  >${log_path}/seq2seq_depoly) >>${log_path}/seq2seq_deploy 2>&1
print_info $? seq2seq_depoly
}
# 17 pretrained_models
pretrained_models() {
CUDA_VISIBLE_DEVICES=${cudaid2}
cd ${nlp_dir}/examples/text_classification/pretrained_models/
time (python -m paddle.distributed.launch train.py --device gpu  --epochs 2 --save_dir ./checkpoints >${log_path}/pretrained_models_train) >>${log_path}/pretrained_models_train 2>&1
print_info $? pretrained_models_train
time (python export_model.py --params_path=./checkpoints/model_100/model_state.pdparams --output_path=./output >${log_path}/pretrained_models_export) >>${log_path}/pretrained_models_export 2>&1
print_info $? pretrained_models_export
time (python deploy/python/predict.py --model_dir=./output >${log_path}/pretrained_models_deploy) >>${log_path}/pretrained_models_deploy 2>&1
print_info $? pretrained_models_deploy
}
# 18 word_embedding 5min
word_embedding(){
CUDA_VISIBLE_DEVICES=${cudaid1}
cd ${nlp_dir}/examples/word_embedding/
# 使用paddlenlp.embeddings.TokenEmbedding
time (python train.py --device='gpu' \
                --lr=5e-4 \
                --batch_size=32 \
                --epochs=1 \
                --use_token_embedding=True \
                --vdl_dir='./vdl_paddlenlp_dir'  >${log_path}/word_embedding_paddlenlp_train) >>${log_path}/word_embedding_paddlenlp_train 2>&1
print_info $? word_embedding_paddlenlp_train
# 使用paddle.nn.Embedding
time (python train.py --device='gpu' \
                --lr=1e-4 \
                --batch_size=32 \
                --epochs=1 \
                --use_token_embedding=False \
                --vdl_dir='./vdl_paddle_dir' >${log_path}/word_embedding_paddle_train) >>${log_path}/word_embedding_paddle_train 2>&1
print_info $? word_embedding_paddle_train
}
# 19 ernie-ctm
ernie-ctm(){
CUDA_VISIBLE_DEVICES=${cudaid2}
cd ${nlp_dir}/examples/text_to_knowledge/ernie-ctm/
cp -r /ssd1/paddlenlp/download/ctm/data ./
time (python -m paddle.distributed.launch  train.py \
    --max_seq_len 128 \
    --batch_size 8   \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --save_steps 15 \
    --output_dir ./tmp/ \
    --device "gpu"   >${log_path}/ernie-ctm_train) >>${log_path}/ernie-ctm_train 2>&1
print_info $? ernie-ctm_train
time (python -u eval.py \
    --max_seq_len 128 \
    --batch_size 8   \
    --init_ckpt_dir ./tmp/ernie_ctm_ft_model_15.pdparams \
    --device "gpu"   >${log_path}/ernie-ctm_eval) >>${log_path}/ernie-ctm_eval 2>&1
print_info $? ernie-ctm_eval
}
# 20 distilbert
distilbert (){
cd /ssd1/paddlenlp/download/distilbert/
rm -rf tmp/
time (python -u ./run_glue_paddle.py \
    --model_type distilbert \
    --seed 250 \
    --model_name_or_path distilbert-base-uncased \
    --task_name mrpc \
    --max_seq_length 128 \
    --batch_size 4   \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --max_steps 1 \
    --logging_steps 1 \
    --save_steps 1 \
    --output_dir ./tmp/mrpc/ \
    --device gpu  >${log_path}/distilbert_train) >>${log_path}/distilbert_train 2>&1
print_info $? distilbert_train
}
# 21 stacl
stacl() {
cd ${nlp_dir}/examples/simultaneous_translation/stacl/
cp -r /ssd1/paddlenlp/download/stacl/* ./
CUDA_VISIBLE_DEVICES=${cudaid2}
time (sed -i "s/save_step: 10000/save_step: 1/g" config/transformer.yaml
sed -i "s/p print_step: 100/print_step: 1/g" config/transformer.yaml
sed -i "s/epoch: 30/epoch: 1/g" config/transformer.yaml
sed -i "s/max_iter: None/max_iter: 3/g" config/transformer.yaml
sed -i "s/batch_size: 4096/batch_size: 500/g" config/transformer.yaml
python -m paddle.distributed.launch train.py --config ./config/transformer.yaml  >${log_path}/stacl_wk-1) >>${log_path}/stacl_wk-1 2>&1
print_info $? stacl_wk-1

time (
sed -i "s/waitk: -1/waitk: 3/g" config/transformer.yaml
sed -i 's/save_model: "trained_models"/save_model: "trained_models_3"/g' config/transformer.yaml
sed -i 's#init_from_checkpoint: ""#init_from_checkpoint: "./trained_models/step_1/"#g' config/transformer.yaml
python -m paddle.distributed.launch  train.py --config ./config/transformer.yaml >${log_path}/stacl_wk3) >>${log_path}/stacl_wk3 2>&1
print_info $? stacl_wk3

time (sed -i "s/waitk: 3/waitk: 5/g" config/transformer.yaml
sed -i 's/save_model: "trained_models_3"/save_model: "trained_models_5"/g' config/transformer.yaml
sed -i 's#init_from_checkpoint: "./trained_models/step_1/"#init_from_checkpoint: "./trained_models_3/step_1/"#g' config/transformer.yaml
python -m paddle.distributed.launch train.py --config ./config/transformer.yaml >${log_path}/stacl_wk5) >>${log_path}/stacl_wk5 2>&1
print_info $? stacl_wk5

time (sed -i "s/batch_size: 500/batch_size: 100/g" config/transformer.yaml
sed -i 's#init_from_params: "trained_models/step_final/"#init_from_params: "./trained_models_5/step_1/"#g' config/transformer.yaml
python predict.py --config ./config/transformer.yaml >${log_path}/stacl_predict) >>${log_path}/stacl_predict 2>&1
print_info $? stacl_predict
}
# 22 transformer
transformer (){
cd ${nlp_dir}/examples/machine_translation/transformer/
CUDA_VISIBLE_DEVICES=${cudaid2}
time (
sed -i "s/save_step: 10000/save_step: 1/g" configs/transformer.base.yaml
sed -i "s/print_step: 100/print_step: 1/g" configs/transformer.base.yaml
sed -i "s/epoch: 30/epoch: 1/g" configs/transformer.base.yaml
sed -i "s/max_iter: None/max_iter: 2/g" configs/transformer.base.yaml
sed -i "s/batch_size: 4096/batch_size: 1000/g" configs/transformer.base.yaml
python -m paddle.distributed.launch train.py --config ./configs/transformer.base.yaml >${log_path}/transformer_train) >>${log_path}/transformer_train 2>&1
print_info $? transformer_train
time (
#predict
sed -i 's#init_from_params: "./trained_models/step/"#init_from_params: "./trained_models/step_1/"#g' configs/transformer.base.yaml
python predict.py --config ./configs/transformer.base.yaml >${log_path}/transformer_predict) >>${log_path}/transformer_predict 2>&1
print_info $? transformer_predict
#export
time (python export_model.py --config ./configs/transformer.base.yaml >${log_path}/transformer_export) >>${log_path}/transformer_export 2>&1
print_info $? transformer_export
#infer
time (cd ./deploy/python/
python inference.py \
        --config ../../configs/transformer.base.yaml \
        --batch_size 8 \
        --device gpu \
        --model_dir ../../infer_model/ >${log_path}/transformer_infer) >>${log_path}/transformer_infer 2>&1
print_info $? transformer_infer
}
# 23 pet
pet (){
cd ${nlp_dir}/examples/few_shot/pet/
CUDA_VISIBLE_DEVICES=${cudaid1}
#chid_train
time (
python  -u -m paddle.distributed.launch  \
    pet.py \
    --task_name "chid" \
    --device gpu \
    --pattern_id 0 \
    --save_dir ./chid \
    --index 0 \
    --batch_size 8 \
    --learning_rate 5E-5 \
    --epochs 1 \
    --max_seq_length 512 \
    --language_model "ernie-1.0"  >${log_path}/pet_chid_train) >>${log_path}/pet_chid_train 2>&1
print_info $? pet_chid_train
#chid_predict
time (
python -u -m paddle.distributed.launch  predict.py \
        --task_name "chid" \
        --device gpu \
        --init_from_ckpt "./chid/model_6/model_state.pdparams" \
        --output_dir "./chid/output" \
        --batch_size 32 \
        --max_seq_length 512 >${log_path}/pet_chid_predict) >>${log_path}/pet_chid_predict 2>&1
print_info $? pet_chid_predict
}
#24 simbert
simbert(){
cd ${nlp_dir}/examples/text_matching/simbert/
cp -r /ssd1/paddlenlp/download/simbert/dev.tsv ./
time (
python predict.py --input_file ./dev.tsv >${log_path}/simbert) >>${log_path}/simbert 2>&1
print_info $? simbert
}
#25 ernie-doc
ernie-doc(){
cd ${nlp_dir}/examples/language_model/ernie-doc/
CUDA_VISIBLE_DEVICES=${cudaid2}
time (python -m paddle.distributed.launch  --log_dir hyp run_classifier.py --epochs 15 --layerwise_decay 0.7 --learning_rate 5e-5 --batch_size 8 --save_steps 2000  --dataset hyp --output_dir hyp >${log_path}/hyp) >>${log_path}/hyp 2>&1
print_info $? hyp
time (python -m paddle.distributed.launch  --log_dir cmrc2018 run_mrc.py --batch_size 8 --layerwise_decay 0.8 --dropout 0.2 --learning_rate 4.375e-5 --epochs 1 --save_steps 5000 --dataset cmrc2018 --output_dir cmrc2018  >${log_path}/cmrc2018) >>${log_path}/cmrc2018 2>&1
print_info $?  cmrc2018
time (python -m paddle.distributed.launch  --log_dir c3 run_mcq.py --learning_rate 6.5e-5 --epochs 1 --save_steps 1000 --output_dir c3 >${log_path}/c3) >>${log_path}/c3 2>&1
print_info $? c3
time (python -m paddle.distributed.launch  --log_dir cail/ run_semantic_matching.py --epochs 1 --layerwise_decay 0.8 --learning_rate 1.25e-5 --batch_size 4  --save_steps 1000 --output_dir cail >${log_path}/cail) >>${log_path}/cail 2>&1
print_info $? cail
time (python -m paddle.distributed.launch  --log_dir msra run_sequence_labeling.py --learning_rate 3e-5 --epochs 1 --save_steps 5000 --output_dir msra  >${log_path}/msar) >>${log_path}/msar 2>&1
print_info $? msar
}
#26 transformer-xl
transformer-xl (){
cd ${nlp_dir}/examples/language_model/transformer-xl/
cp -r /ssd1/paddlenlp/download/transformer-xl/* ./
CUDA_VISIBLE_DEVICES=${cudaid2}
time (sed -i 's/print_step: 100/print_step: 1/g' configs/enwik8.yaml
sed -i 's/save_step: 20000/save_step: 3/g' configs/enwik8.yaml
sed -i 's/max_step: 400000/max_step: 4/g' configs/enwik8.yaml
python3.7 -m paddle.distributed.launch  train.py --config ./configs/enwik8.yaml >${log_path}/train_enwik8) >>${log_path}/train_enwik8 2>&1
print_info $? train_enwik8
time (
sed -i 's#init_from_params: "./trained_models/step_final/"#init_from_params: "./trained_models/step_3/"#g' configs/enwik8.yaml
python eval.py --config ./configs/enwik8.yaml >${log_path}/eval_enwik8) >>${log_path}/eval_enwik8 2>&1
print_info $? eval_enwik8
}
#27 pointer_summarizer
pointer_summarizer() {
cd ${nlp_dir}/examples/text_summarization/pointer_summarizer/
cp -r /ssd1/paddlenlp/download/pointer_summarizer/* ./
CUDA_VISIBLE_DEVICES=${cudaid1}
time (sed -i 's/max_iterations = 100000/max_iterations = 5/g' config.py
sed -i 's/if iter % 5000 == 0 or iter == 1000:/if iter % 5 == 0 :/g' train.py
python train.py >${log_path}/pointer_summarizer_train) >>${log_path}/pointer_summarizer_train 2>&1
print_info $? pointer_summarizer_train
}
####################################
export P0case_list=()
export P0case_time=0
export all_P0case_time=0
declare -A all_P0case_dic
all_P0case_dic=(["waybill_ie"]=3 ["msra_ner"]=15 ["glue"]=2 ["bert"]=2 ["skep"]=10 ["bigbird"]=2 ["electra"]=2  ["gpt"]=2 ["xlnet"]=2 \
 ["ofa"]=2 ["albert"]=2   ["squad"]=20 ["tinybert"]=5 ["lexical_analysis"]=5 ["seq2seq"]=5 ["pretrained_models"]=10 ["word_embedding"]=5 \
  ["ernie-ctm"]=5 ["distilbert"]=5  ["stacl"]=5 ["transformer"]=5 ["pet"]=5 ["simbert"]=5 ["ernie-doc"]=20 ["transformer-xl"]=5 ["pointer_summarizer"]=5)
get_diff_TO_P0case(){
for key in $(echo ${!all_P0case_dic[*]});do
    all_P0case_time=`expr ${all_P0case_time} + ${all_P0case_dic[$key]}`
done
P0case_list=(waybill_ie msra_ner glue bert skep bigbird electra gpt xlnet ofa albert squad tinybert lexical_analysis seq2seq \
pretrained_models word_embedding ernie-ctm distilbert stacl transformer pet simbert ernie-doc transformer-xl pointer_summarizer)
P0case_time=${all_P0case_time}
}
set -e
get_diff_TO_P0case
echo -e "\033[35m ---- P0case_list length: ${#P0case_list[*]}, cases: ${P0case_list[*]} \033[0m"
echo -e "\033[35m ---- P0case_time: $P0case_time min \033[0m"
set +e
####################################
echo -e "\033[35m ---- start run P0case  \033[0m"
case_num=1
for p0case in ${P0case_list[*]};do
    echo -e "\033[35m ---- running P0case $case_num/${#P0case_list[*]}: ${p0case} \033[0m"
    ${p0case}
    let case_num++
done
echo -e "\033[35m ---- end run P0case  \033[0m"

cd ${nlp_dir}/logs
FF=`ls *_FAIL*|wc -l`
if [ "${FF}" -gt "0" ];then
    exit 1
else
    exit 0
fi
