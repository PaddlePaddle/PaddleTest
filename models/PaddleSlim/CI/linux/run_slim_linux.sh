#!/bin/bash
export repo_path=$PWD
##$1:cudaid1 $2:cudaid2 $3:proxy $4:slim_branch $5:det_data_path $6:run_CI/run_ALL

cudaid1=$1
cudaid2=$2
export CUDA_VISIBLE_DEVICES=${cudaid1}
echo ---cudaid1:${cudaid1}, cudaid2:${cudaid2}---

export https_proxy=$3
export http_proxy=$3
export no_proxy=localhost,bj.bcebos.com,su.bcebos.com

export det_data_path=$5
echo ----${det_data_path}----

print_info(){
if [ $1 -ne 0 ];then
    mv ${log_path}/$2 ${log_path}/FAIL_$2.log
    echo -e "\033[31m ${log_path}/FAIL_$2 \033[0m"
    echo "fail log as follow"
    cat  ${log_path}/FAIL_$2.log
else
    mv ${log_path}/$2 ${log_path}/SUCCESS_$2.log
    echo -e "\033[32m ${log_path}/SUCCESS_$2 \033[0m"
fi
}

###################
echo --------- git repo -----
git clone https://github.com/PaddlePaddle/PaddleSlim.git -b $4
git clone https://github.com/PaddlePaddle/PaddleClas.git -b develop
git clone https://github.com/PaddlePaddle/PaddleDetection.git -b develop
git clone https://github.com/PaddlePaddle/PaddleOCR.git -b dygraph
git clone https://github.com/PaddlePaddle/PaddleNLP.git -b develop
git clone https://github.com/PaddlePaddle/PaddleSeg.git -b develop

echo --------- repo list -----
ls
################

if [ -d "$PWD/logs" ];then
    rm -rf $PWD/logs;
fi
mkdir $PWD/logs
export all_log_path=$PWD/logs

echo --------- env variable-----
env

echo -------start install paddleslim----
cd ${repo_path}/PaddleSlim
python -m pip install -r requirements.txt
python setup.py install
echo ------finish install paddleslim -----
python -m pip list | grep paddleslim

slim_ocr_prune_MobileNetV3(){
	cd ${repo_path}/PaddleOCR
    wget -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/pretrained/MobileNetV3_large_x0_5_pretrained.pdparams
	python deploy/slim/prune/sensitivity_anal.py -c configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml \
-o Global.pretrained_model=./pretrain_models/MobileNetV3_large_x0_5_pretrained \
Global.save_model_dir=./output/prune_model \
Global.epoch_num=1 > ${log_path}/slim_ocr_prune_MobileNetV3 2>&1

print_info $? slim_ocr_prune_MobileNetV3

# export model 不依赖paddleslim
# pytho3.7 deploy/slim/prune/export_prune_model.py
}

slim_ocr_quant_ocr_mobile_v2(){
	cd ${repo_path}/PaddleOCR
    wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_train.tar
    tar -xf ch_ppocr_mobile_v2.0_det_train.tar
	python deploy/slim/quantization/quant.py -c configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml \
-o Global.pretrained_model=./ch_ppocr_mobile_v2.0_det_train/best_accuracy \
 Global.save_model_dir=./output/quant_inference_model \
 Global.epoch_num=1 > ${log_path}/slim_ocr_quant_ocr_mobile_v2 2>&1

print_info $? slim_ocr_quant_ocr_mobile_v2
}

# V100 16G 会OOM、设置batch_size_per_card=4 可解决；
slim_ocr_quant_distill_mobile_v2(){
    cd ${repo_path}/PaddleOCR
    wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_distill_train.tar
    tar xf ch_PP-OCRv3_det_distill_train.tar
    python deploy/slim/quantization/quant.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml \
-o Global.pretrained_model='./ch_PP-OCRv3_det_distill_train/best_accuracy' \
Global.save_model_dir=./output/quant_model_distill \
Global.epoch_num=1 \
Train.loader.batch_size_per_card=4 > ${log_path}/slim_ocr_quant_distill_mobile_v2 2>&1

print_info $? slim_ocr_quant_distill_mobile_v2
}


slim_ocr(){
mkdir ${all_log_path}/slim_ocr_log
export log_path=${all_log_path}/slim_ocr_log
cd ${repo_path}/PaddleOCR
python -m pip install -r requirements.txt

#准备数据
wget -nc -P ./train_data/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/test/icdar2015.tar
cd ./train_data/ && tar xf icdar2015.tar && cd ../

if [ "$1" == "run_CI" ];then
    slim_ocr_prune_MobileNetV3
    slim_ocr_quant_ocr_mobile_v2
elif [ "$1" == "run_ALL" ];then
    slim_ocr_prune_MobileNetV3
    slim_ocr_quant_best_accuracy
    slim_ocr_quant_distill_mobile_v2
else
    echo ---only run_CI or run_ALL---
fi

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

slim_nlp_distill_minilmv2(){
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
    --max_steps 10 \
    --output_dir ./tmp/$TASK_NAME/  \
    --device gpu > ${log_path}/slim_nlp_bert_Finetuning 2>&1
print_info $? slim_nlp_bert_Finetuning 

    cd ${repo_path}/PaddleNLP/examples/model_compression/ofa/
python -u ./run_glue_ofa.py --model_type bert \
    --model_name_or_path ../../benchmark/glue/tmp/SST-2/sst-2_ft_model_10.pdparams \
    --task_name $TASK_NAME \
    --max_seq_length 128    \
    --batch_size 32       \
    --learning_rate 2e-5     \
    --num_train_epochs 1     \
    --logging_steps 10     \
    --save_steps 50     \
    --output_dir ./tmp/$TASK_NAME \
    --device gpu  \
    --max_steps 200 \
    --width_mult_list 1.0 0.8333333333333334 0.6666666666666666 0.5 > ${log_path}/slim_nlp_bert_ofa 2>&1
print_info $? slim_nlp_bert_ofa 

}


slim_nlp_prune_quant_pp_minilm(){
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

export MODEL_PATH=pruned_models
export TASK_NAME=CLUEWSC2020
sh export.sh ${MODEL_PATH} ${TASK_NAME} > ${log_path}/nlp_pp_minilm_export_model_after_prune 2>&1
print_info $? nlp_pp_minilm_export_model_after_prune

cd ../quantization/
export MODEL_DIR=../pruning/pruned_models/
python quant_post.py --task_name $TASK_NAME --input_dir ${MODEL_DIR}/${TASK_NAME}/0.75/sub_static > ${log_path}/slim_nlp_pp_minilm_quant 2>&1
print_info $? slim_nlp_pp_minilm_quant
}

slim_nlp(){
mkdir ${all_log_path}/slim_nlp_log
export log_path=${all_log_path}/slim_nlp_log
cd ${repo_path}/PaddleNLP
python -m pip install -r requirements.txt
python setup.py install

if [ "$1" == "run_CI" ];then
    slim_nlp_prune_quant_pp_minilm
elif [ "$1" == "run_ALL" ];then
    slim_nlp_ofa_bert
    slim_nlp_prune_quant_pp_minilm
else
    echo ---only run_CI or run_ALL---
fi

# distill 不依赖paddleslim
#slim_nlp_distill_minilmv2
#slim_nlp_distill_lstm
}

slim_det_prune_yolov3_mv1(){
	cd ${repo_path}/PaddleDetection
	python tools/train.py -c configs/yolov3/yolov3_mobilenet_v1_270e_voc.yml -o  epoch=1 TrainReader.batch_size=8 \
	--slim_config configs/slim/prune/yolov3_prune_l1_norm.yml > ${log_path}/slim_det_prune_yolov3_mv1 2>&1
print_info $? slim_det_prune_yolov3_mv1
}

slim_det_post_quant_ppyolo_mbv3(){
	cd ${repo_path}/PaddleDetection
	python tools/post_quant.py -c configs/ppyolo/ppyolo_mbv3_large_coco.yml \
    --slim_config=configs/slim/post_quant/ppyolo_mbv3_large_ptq.yml > ${log_path}/slim_det_post_quant_ppyolo_mbv3 2>&1
print_info $? slim_det_post_quant_ppyolo_mbv3
}

slim_det_pact_quant_ppyolo_r50vd(){
    cd ${repo_path}/PaddleDetection
    python tools/train.py -c configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml \
    -o epoch=1 TrainReader.batch_size=12 > ${log_path}/slim_det_pact_quant_ppyolo_r50vd 2>&1
print_info $? slim_det_pact_quant_ppyolo_r50vd
}

slim_det_normal_quant_ppyolo_mbv3(){
    cd ${repo_path}/PaddleDetection
    python tools/train.py -c configs/ppyolo/ppyolo_mbv3_large_coco.yml \
    -o epoch=1 TrainReader.batch_size=12 > ${log_path}/slim_det_normal_quant_ppyolo_mbv3 2>&1
print_info $? slim_det_normal_quant_ppyolo_mbv3
}

slim_det_prune_distill_yolov3_mv1(){
    cd ${repo_path}/PaddleDetection
    python tools/train.py -c configs/yolov3/yolov3_mobilenet_v1_270e_coco.yml -o  epoch=1 TrainReader.batch_size=8 \
  --slim_config configs/slim/extensions/yolov3_mobilenet_v1_coco_distill_prune.yml > ${log_path}/slim_det_prune_distill_yolov3_mv1 2>&1
print_info $? slim_det_prune_distill_yolov3_mv1
}

slim_det_nas_blazeface(){
    cd ${repo_path}/PaddleDetection/tree/develop/static/slim/nas
    python -u train_nas.py -c blazeface.yml -o max_iters=10 search_steps=1  > ${log_path}/slim_det_nas_blazeface 2>&1
print_info $? slim_det_nas_blazeface
}

slim_detection(){
mkdir ${all_log_path}/slim_detection_log
export log_path=${all_log_path}/slim_detection_log
cd ${repo_path}/PaddleDetection
python -m pip install -U pip Cython
python -m pip install -r requirements.txt

cd dataset/voc/
ln -s ${det_data_path}/pascalvoc/trainval.txt trainval.txt
ln -s ${det_data_path}/pascalvoc/test.txt test.txt
ln -s ${det_data_path}/pascalvoc/VOCdevkit VOCdevkit

cd ../coco/
ln -s ${det_data_path}/coco/val2017 val2017
ln -s ${det_data_path}/coco/annotations annotations
ln -s ${det_data_path}/coco/train2017 train2017

if [ "$1" == "run_CI" ];then
    slim_det_prune_yolov3_mv1
    slim_det_post_quant_ppyolo_mbv3
elif [ "$1" == "run_ALL" ];then
    slim_det_prune_yolov3_mv1
    slim_det_post_quant_ppyolo_mbv3
    slim_det_pact_quant_ppyolo_r50vd
    slim_det_normal_quant_ppyolo_mbv3
    # 存在bug、需det repo修复、5.11记录
    #slim_det_prune_distill_yolov3_mv1
    # 存在bug、需slim repo提pr修复、5.11记录
    #slim_det_nas_blazeface
else
    echo ---only run_CI or run_ALL---
fi
}

slim_clas_quant_ResNet50_vd_gpu2(){
	cd ${repo_path}/PaddleClas
	export CUDA_VISIBLE_DEVICES=${cudaid2}
	python -m paddle.distributed.launch \
    tools/train.py \
    -c ppcls/configs/slim/ResNet50_vd_quantization.yaml \
	-o Global.epochs=1 \
	-o DataLoader.Train.sampler.batch_size=32 > ${log_path}/slim_clas_quant_ResNet50_vd_gpu2 2>&1
print_info $? slim_clas_quant_ResNet50_vd_gpu2
}

slim_clas_quant_ResNet50_vd_gpu1(){
    cd ${repo_path}/PaddleClas
    export CUDA_VISIBLE_DEVICES=${cudaid2}
    python tools/train.py \
    -c ppcls/configs/slim/ResNet50_vd_quantization.yaml \
    -o Global.epochs=1 \
    -o DataLoader.Train.sampler.batch_size=32 > ${log_path}/slim_clas_quant_ResNet50_vd_gpu1 2>&1
print_info $? slim_clas_quant_ResNet50_vd_gpu1
}

slim_clas_post_quant_ResNet50_vd(){
	cd ${repo_path}/PaddleClas
	wget -P ./cls_pretrain/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/ResNet50_vd_pretrained.pdparams

    python tools/export_model.py -c ./ppcls/configs/ImageNet/ResNet/ResNet50_vd.yaml   \
	-o Global.pretrained_model=./cls_pretrain/ResNet50_vd_pretrained   \
	-o Global.save_inference_dir=./deploy/models/class_ResNet50_vd_ImageNet_infer > ${log_path}/clas_export_model_ResNet50_vd 2>&1
print_info $? clas_export_model_ResNet50_vd	

	python deploy/slim/quant_post_static.py -c ppcls/configs/ImageNet/ResNet/ResNet50_vd.yaml \
	 -o Global.save_inference_dir=./deploy/models/class_ResNet50_vd_ImageNet_infer \
	 -o Global.epochs=1 \
	 -o Global.pretrained_model=./cls_pretrain/ResNet50_vd_pretrained > ${log_path}/slim_clas_post_quant_ResNet50_vd 2>&1
print_info $? slim_clas_post_quant_ResNet50_vd
}

slim_clas_prune_ResNet50_vd_gpu2(){
	cd ${repo_path}/PaddleClas
	export CUDA_VISIBLE_DEVICES=${cudaid2}
	python -m paddle.distributed.launch \
    tools/train.py \
    -c ppcls/configs/slim/ResNet50_vd_prune.yaml \
	-o Global.epochs=1 > ${log_path}/slim_clas_prune_ResNet50_vd_gpu2 2>&1
print_info $? slim_clas_prune_ResNet50_vd_gpu2
}

slim_clas_prune_ResNet50_vd_gpu1(){
    cd ${repo_path}/PaddleClas
    python tools/train.py \
    -c ppcls/configs/slim/ResNet50_vd_prune.yaml \
    -o Global.epochs=1 > ${log_path}/slim_clas_prune_ResNet50_vd_gpu1 2>&1
print_info $? slim_clas_prune_ResNet50_vd_gpu1
}

slim_clas(){
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

if [ "$1" == "run_CI" ];then
    slim_clas_quant_ResNet50_vd_gpu2
    slim_clas_prune_ResNet50_vd_gpu2
elif [ "$1" == "run_ALL" ];then
    slim_clas_quant_ResNet50_vd_gpu2
    slim_clas_post_quant_ResNet50_vd
    slim_clas_quant_ResNet50_vd_gpu1
    slim_clas_prune_ResNet50_vd_gpu2
    slim_clas_prune_ResNet50_vd_gpu1
else
    echo ---only run_CI or run_ALL---
fi
}

slim_seg_quant_BiseNetV2(){
    cd ${repo_path}/PaddleSeg
    python train.py --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml \
    --do_eval  --use_vdl  --save_interval 250 --save_dir output_fp32 \
    --iters 100 > ${log_path}/seg_BiseNetV2_output 2>&1
print_info $? seg_BiseNetV2_output

    python slim/quant/qat_train.py \
    --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml \
    --model_path output_fp32/best_model/model.pdparams \
    --learning_rate 0.001 --do_eval --use_vdl \
    --save_interval 250 --save_dir output_quant \
    --iters 100 ${log_path}/slim_seg_quant_BiseNetV2 2>&1
print_info $? slim_seg_quant_BiseNetV2
}

slim_seg_prune_BiseNetV2(){
    cd ${repo_path}/PaddleSeg
    python train.py --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml \
    --do_eval --use_vdl --save_interval 500 --save_dir \
    output --iters 100 > ${log_path}/seg_BiseNetV2_output_prune 2>&1
print_info $? seg_BiseNetV2_output_prune

    python slim/prune/prune.py  --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml \
     --pruning_ratio 0.2  --model_path output/best_model/model.pdparams  \
     --retraining_iters 100   --save_dir prune_model ${log_path}/slim_seg_prune_BiseNetV2 2>&1
print_info $? slim_seg_prune_BiseNetV2
}

slim_seg(){
    mkdir ${all_log_path}/slim_seg_log
    export log_path=${all_log_path}/slim_seg_log
    cd ${repo_path}/PaddleSeg
    python -m pip install -r requirements.txt
    python setup.py install

if [ "$1" == "run_CI" ];then
    slim_seg_quant_BiseNetV2
    #slim_seg_prune_BiseNetV2
elif [ "$1" == "run_ALL" ];then
    slim_seg_quant_BiseNetV2
    # 存在bug、5.11记录
    #slim_seg_prune_BiseNetV2
else
    echo ---only run_CI or run_ALL---
fi

}

echo -------start run case----

slim_detection $6

slim_ocr $6

slim_nlp $6

slim_clas $6

slim_seg $6

echo -------finish run case----

echo --------fail logs numbers ------
FF=$(ls ${all_log_path}/slim_* | grep -i fail |  wc -l)

if [ ${FF} -gt 0 ];then
    echo -----fail case:${FF}----
    ls ${all_log_path}/slim_* | grep -i fail
    echo -------------failed----------
    exit ${FF};
else
    echo -----fail case:${FF}----
    echo -------------passed----------
    exit 0;
fi  
