#!/bin/bash
export repo_path=$PWD
##$1:cudaid1 $2:cudaid2 $3:proxy $4:slim_branch

cudaid1=$1
cudaid2=$2
export CUDA_VISIBLE_DEVICES=${cudaid1}
echo -------cudaid1:${cudaid1}, cudaid2:${cudaid2}---

export https_proxy=$3
export http_proxy=$3
export no_proxy=localhost,bj.bcebos.com,su.bcebos.com

print_info(){
if [ $1 -ne 0 ];then
    mv ${log_path}/$2 ${log_path}/FAIL_$2.log
    echo -e "\033[31m ${log_path}/FAIL_$2 \033[0m"
    echo "fail log as follow"
    cat  ${log_path}/FAIL_$2.log
else
    mv ${log_path}/$2 ${log_path}/SUCCESS_$2.log
    echo -e "\033[32m ${log_path}/SUCCESS_$2 \033[0m"
    cat  ${log_path}/SUCCESS_$2.log
fi
}

###################
echo --------- git repo -----
git clone https://github.com/PaddlePaddle/PaddleSlim.git -b $4
git clone https://github.com/PaddlePaddle/PaddleClas.git -b develop
git clone https://github.com/PaddlePaddle/PaddleDetection.git -b develop
git clone https://github.com/PaddlePaddle/PaddleOCR.git -b dygraph
git clone https://github.com/PaddlePaddle/PaddleNLP.git -b develop

echo --------- repo list -----
ls
################

if [ -d "$PWD/logs" ];then
    rm -rf $PWD/logs;
fi
mkdir $PWD/logs
touch $PWD/logs/result.log
export all_log_path=$PWD/logs

echo --------- env variable-----
env

echo -------start install paddleslim----
cd ${repo_path}/PaddleSlim
python -m pip install -r requirements.txt
python setup.py install
echo ------finish install paddleslim -----
python -m pip list | grep paddleslim

echo ------slim ocr------

slim_ocr_prune_MobileNetV3{
	cd ${repo_path}/PaddleOCR
	python deploy/slim/prune/sensitivity_anal.py -c configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml \
-o Global.pretrained_model=./pretrain_models/MobileNetV3_large_x0_5_pretrained \
Global.save_model_dir=./output/prune_model/ Global.epoch_num=1 > ${log_path}/slim_ocr_prune_MobileNetV3 2>&1

print_info $? slim_ocr_prune_MobileNetV3
}

slim_ocr_quant_best_accuracy{
	cd ${repo_path}/PaddleOCR
	python deploy/slim/quantization/quant.py -c configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml \
-o Global.pretrained_model=./ch_ppocr_mobile_v2.0_det_train/best_accuracy \
 Global.save_model_dir=./output/quant_inference_model Global.epoch_num=1 > ${log_path}/slim_ocr_quant_best_accuracy 2>&1

print_info $? slim_ocr_quant_best_accuracy
}


slim_ocr(){
mkdir ${all_log_path}/slim_ocr_log
export log_path=${all_log_path}/slim_ocr_log
cd ${repo_path}/PaddleOCR
python -m pip install -r requirements.txt

#数据准备
wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/icdar2015.ta
cd ./train_data/ && tar xf icdar2015.tar && cd ../
#预训练模型
wget -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/pretrained/MobileNetV3_large_x0_5_pretrained.pdparams

wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_train.tar
tar -xf ch_ppocr_mobile_v2.0_det_train.tar

slim_ocr_prune_MobileNetV3
slim_ocr_quant_best_accuracy
}

slim_nlp_distill_lstm(){
	cd ${repo_path}/PaddleNLP/examples/model_compression/distill_lstm
	wget https://bj.bcebos.com/paddlenlp/data/senta_word_dict.txt
	wget https://paddle-qa.bj.bcebos.com/PaddleSlim_datasets/best_model_610.tar.gz
    tar -xvf best_model_610.tar.gz
python small.py \
    --task_name sst-2 \
    --vocab_size 30522 \
    --max_epoch 1 \
    --batch_size 64 \
    --lr 1.0 \
    --dropout_prob 0.4 \
    --output_dir small_models/SST-2 \
    --save_steps 1000 \
    --embedding_name w2v.google_news.target.word-word.dim300.en > ${log_path}/slim_nlp_distill_lstm_sst2_small 2>&1
print_info $? slim_nlp_distill_lstm_sst2_small

python bert_distill.py \
    --task_name sst-2 \
    --vocab_size 30522 \
    --max_epoch 1 \
    --lr 1.0 \
    --task_name sst-2 \
    --dropout_prob 0.2 \
    --batch_size 128 \
    --model_name bert-base-uncased \
    --output_dir distilled_models/SST-2 \
    --teacher_dir best_model_610 \
    --save_steps 1000 \
    --embedding_name w2v.google_news.target.word-word.dim300.en > ${log_path}/slim_nlp_distill_lstm_sst2_distill 2>&1
print_info $? slim_nlp_distill_lstm_sst2_distill 
}

slim_nlp_distill_minilmv2{
	cd ${repo_path}/PaddleNLP//model_compression/minilmv2/
	wget https://paddlenlp.bj.bcebos.com/models/general_distill/minilmv2_6l_768d_ch.tar.gz
    tar -zxf minilmv2_6l_768d_ch.tar.gz

    python -m paddle.distributed.launch --gpus ${cudaid2} general_distill.py \
  --student_model_type tinybert \
  --num_relation_heads 48 \
  --student_model_name_or_path tinybert-6l-768d-zh \
  --init_from_student False \
  --teacher_model_type bert \
  --teacher_model_name_or_path bert-base-chinese \
  --max_seq_length 128 \
  --batch_size 256 \
  --learning_rate 6e-4 \
  --logging_steps 10 \
  --max_steps 30 \
  --warmup_steps 4000 \
  --save_steps 10 \
  --teacher_layer_index 11 \
  --student_layer_index 5 \
  --weight_decay 1e-2 \
  --output_dir ./pretrain \
  --input_dir ./minilmv2_6l_768d_ch > ${log_path}/slim_nlp_distill_minilmv2 2>&1
print_info $? slim_nlp_distill_minilmv2 
}

slim_nlp_ofa_bert(){
	cd ${repo_path}/PaddleNLP/examples/model_compression/ofa/
	cd ../../benchmark/glue/
    export TASK_NAME=SST-2
python -u run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --max_seq_length 128 \
    --batch_size 32   \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --save_steps 500 \
    --max_steps 100 \
    --output_dir ./tmp/$TASK_NAME/  \
    --device gpu > ${log_path}/slim_nlp_bert_Finetuning 2>&1
print_info $? slim_nlp_bert_Finetuning 

    cd ${repo_path}/PaddleNLP/examples/model_compression/ofa/
python -u ./run_glue_ofa.py --model_type bert \
          --model_name_or_path ../../benchmark/glue/tmp/SST-2/sst-2_ft_model_10.pdparams \
          --task_name $TASK_NAME \
          --max_seq_length 128     \
          --batch_size 32       \
          --learning_rate 2e-5     \
          --num_train_epochs 6     \
          --logging_steps 10     \
          --save_steps 50     \
          --output_dir ./tmp/$TASK_NAME \
          --device gpu  \
          --max_steps 200 \
          --width_mult_list 1.0 0.8333333333333334 0.6666666666666666 0.5 > ${log_path}/slim_nlp_bert_ofa 2>&1
print_info $? slim_nlp_bert_ofa 

}


slim_nlp_pp_minilm(){
	cd ${repo_path}/PaddleNLP/examples/model_compression/pp-minilm/
	cd finetuning
sh run_clue.sh CLUEWSC2020 1e-4 32 1 128 0 ppminilm-6l-768h > ${log_path}/nlp_pp_minilm_finetuning 2>&1
print_info $? nlp_pp_minilm_finetuning

export TASK_NAME=CLUEWSC2020
export MODEL_PATH=ppminilm-6l-768h
export LR=1e-4
export BS=32
python export_model.py --task_name ${TASK_NAME} --model_path ${MODEL_PATH}/models/${TASK_NAME}/${LR}_${BS}/ > ${log_path}/nlp_pp_minilm_export_model 2>&1
print_info $? nlp_pp_minilm_export_model

cd ../pruning/
export FT_MODELS=../finetuning/ppminilm-6l-768h/models/CLUEWSC2020/1e-4_32
sh prune.sh CLUEWSC2020 1e-4 32 1 128 0 ${FT_MODELS} 0.75 > ${log_path}/slim_nlp_pp_minilm_prune 2>&1
print_info $? slim_nlp_pp_minilm_prune

export TASK_NAME=CLUEWSC2020
sh export.sh ${MODEL_PATH} ${TASK_NAME} > ${log_path}/nlp_pp_minilm_export_model_after_prune 2>&1
print_info $? nlp_pp_minilm_export_model_after_prune

cd ../quantization/
export MODEL_DIR=../pruning/pruned_models/
python quant_post.py --task_name $TASK_NAME --input_dir ${MODEL_DIR}/${TASK_NAME}/0.75/sub_static > ${log_path}/nlp_pp_minilm_quant 2>&1
print_info $? nlp_pp_minilm_quant
}

slim_nlp(){
mkdir ${all_log_path}/slim_nlp_log
export log_path=${all_log_path}/slim_nlp_log
cd ${repo_path}/PaddleNLP
python -m pip install -r requirements.txt
python setup.py install

#slim_nlp_distill_minilmv2
#slim_nlp_distill_lstm
slim_nlp_ofa_bert
slim_nlp_pp_minilm
}

slim_det_prune(){
	cd ${repo_path}/PaddleDetection
	python tools/train.py -c configs/yolov3/yolov3_mobilenet_v1_270e_voc.yml -o  epoch=1  batch_size=1 \
	--slim_config configs/slim/prune/yolov3_prune_l1_norm.yml > ${log_path}/slim_det_prune 2>&1
print_info $? slim_det_prune
}

slim_det_quant(){
	cd ${repo_path}/PaddleDetection
	python tools/post_quant.py -c configs/ppyolo/ppyolo_mbv3_large_coco.yml \
	 --slim_config=configs/slim/post_quant/ppyolo_mbv3_large_ptq.yml > ${log_path}/slim_det_quant 2>&1
 > ${log_path}/slim_det_prune 2>&1
print_info $? slim_det_quant
}

slim_detection(){
mkdir ${all_log_path}/slim_detection_log
export log_path=${all_log_path}/slim_detection_log
cd ${repo_path}/PaddleDetection
python -m pip install -r requirements.txt

slim_det_prune
#slim_det_quant
}

slim_clas_quant{
	cd ${repo_path}/PaddleClas
	python -m paddle.distributed.launch \
    --gpus=${cudaid2} tools/train.py \
     -c ppcls/configs/slim/ResNet50_vd_quantization.yaml \
	 -o Global.epochs=1 \
	 -o DataLoader.Train.sampler.batch_size=32 > ${log_path}/slim_clas_quant 2>&1
print_info $? slim_clas_quant
}

slim_clas_post_quant{
	cd ${repo_path}/PaddleClas
	wget -P ./cls_pretrain/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet50_vd_pretrained.pdparams

	python deploy/slim/quant_post_static.py -c ppcls/configs/ImageNet/ResNet/ResNet50_vd.yaml \
	 -o Global.save_inference_dir=./deploy/models/class_ResNet50_vd_ImageNet_infer \
	 -o Global.epochs=1 \
	 -o Global.pretrained_model=./cls_pretrain/ResNet50_vd_pretrained > ${log_path}/slim_clas_post_quant 2>&1
print_info $? slim_clas_post_quant

}

slim_clas_prune{
	cd ${repo_path}/PaddleClas
	python -m paddle.distributed.launch \
    --gpus=${cudaid2} tools/train.py \
    -c ppcls/configs/slim/ResNet50_vd_prune.yaml \
	-o Global.epochs=1 > ${log_path}/slim_clas_prune 2>&1
print_info $? slim_clas_prune
}

slim_clas{
	mkdir ${all_log_path}/slim_clas_log
    export log_path=${all_log_path}/slim_clas_log
    cd ${repo_path}/PaddleClas
    python -m pip install -r requirements.txt

cd dataset
rm -rf ILSVRC2012
wget -nc https://paddle-imagenet-models-name.bj.bcebos.com/data/whole_chain/whole_chain_CIFAR100.tar
tar xf whole_chain_CIFAR100.tar
ln -s whole_chain_CIFAR100 ILSVRC2012
cd ILSVRC2012

mv train.txt train_list.txt
mv test.txt val_list.txt
cd ../..

slim_clas_post_quant
slim_clas_quant
slim_clas_prune
}

echo -------start run case----

echo -------start run ocr----
slim_ocr
echo -------start run nlp----
slim_nlp
echo -------start run clas----
slim_clas
echo -------start run detection----
#slim_detection

echo -------finish run case----

echo --------fail logs numbers ------
FF=$(ls ${all_log_path}/slim_* | grep -i fail |  wc -l)

if [ ${FF} -gt 0 ];then
    echo -----fail case:${FF}----
    echo -------------failed----------
    exit 1;
else
    echo -----fail case:${FF}----
    echo -------------passed----------
    exit 0;
fi  

