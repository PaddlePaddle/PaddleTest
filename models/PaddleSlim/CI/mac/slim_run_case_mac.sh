#!/usr/bin/env bash
##################
echo --- run slim case of mac ---

print_info(){
if [ $1 -ne 0 ];then
    mv ${log_path}/$2 ${log_path}/FAIL_$2.log
    echo -e "\033[31m ${log_path}/FAIL_$2 \033[0m"
    echo "fail log as belows"
    cat ${log_path}/FAIL_$2.log
else
    mv ${log_path}/$2 ${log_path}/SUCCESS_$2.log
    echo -e "\033[32m ${log_path}/SUCCESS_$2 \033[0m"
fi
}

catchException() {
  echo $1 failed due to exception >> FAIL_Exception.log
}


# 1 dist
demo_distillation(){
cd ${slim_dir}/demo/distillation || catchException demo_distillation
if [ -d "output" ];then
    rm -rf output
fi
python distill.py --num_epochs 1 --batch_size 64 --save_inference True --use_gpu False \
--model ResNet50 --teacher_model ResNet101_vd \
--teacher_pretrained_model ../pretrain/ResNet101_vd_pretrained >${log_path}/distill_ResNet101_vd_ResNet50_T 2>&1
print_info $? distill_ResNet101_vd_ResNet50_T
}

all_distillation(){
    demo_distillation
}

demo_st_quant_post(){
cd ${slim_dir}/demo/quant/quant_post || catchException demo_st_quant_post
pwd
wget -P inference_model https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV1_infer.tar
cd inference_model/
tar -xf MobileNetV1_infer.tar
cd ..
pwd

for algo in hist
do
# 带bc参数、指定algo、且为新存储格式的离线量化
python quant_post.py --model_path ./inference_model/MobileNetV1_infer/ \
--save_path ./quant_model/${algo}_bc/MobileNet \
--model_filename model --params_filename weights \
--algo ${algo} --bias_correction True --use_gpu False >${log_path}/st_quant_post_T_${algo}_bc 2>&1
print_info $? st_quant_post_T_${algo}_bc

# 量化后eval
echo "quant_post eval bc " ${algo}
python eval.py --model_path ./quant_model/${algo}_bc/MobileNet --model_name __model__ \
--params_name __params__ --use_gpu False > ${log_path}/st_quant_post_${algo}_bc_eval 2>&1
print_info $? st_quant_post_${algo}_bc_eval

done
}

demo_st_pact_quant_aware(){
cd ${slim_dir}/demo/quant/pact_quant_aware
python train.py --model MobileNetV3_large_x1_0 \
--pretrained_model ../../pretrain/MobileNetV3_large_x1_0_ssld_pretrained \
--num_epochs 1 --lr 0.0001 --use_pact True --batch_size 128 --lr_strategy=piecewise_decay \
--step_epochs 2 --l2_decay 1e-5 --use_gpu False >${log_path}/demo_quant_pact_quant_aware_v3 2>&1
print_info $? demo_quant_pact_quant_aware_v3
}

demo_dygraph_quant(){
cd ${slim_dir}/demo/dygraph/quant
python train.py --model='mobilenet_v1' \
--pretrained_model '../../pretrain/MobileNetV1_pretrained' \
--num_epochs 1 \
--batch_size 128 --use_gpu False > ${log_path}/dy_quant_v1 2>&1
print_info $? dy_quant_v1
}

demo_st_quant_aware(){
cd ${slim_dir}/demo/quant/quant_aware || catchException demo_st_quant_aware
python train.py --model MobileNet \
--pretrained_model ../../pretrain/MobileNetV1_pretrained \
--checkpoint_dir ./output/mobilenetv1 --num_epochs 1 --batch_size 128 --use_gpu False >${log_path}/quant_aware_v1_T 2>&1
print_info $? quant_aware_v1_T
}

all_quant(){
    demo_st_quant_aware
    demo_st_quant_post
    demo_st_pact_quant_aware
    demo_dygraph_quant
}


demo_prune_v1(){
cd ${slim_dir}/demo/prune
if [ -d "models" ];then
    rm -rf models
fi
python train.py --model "MobileNet" --pruned_ratio 0.31 --data "imagenet" \
--pretrained_model ../pretrain/MobileNetV1_pretrained/ --num_epochs 1 --use_gpu False >${log_path}/prune_v1_T 2>&1
print_info $? prune_v1_T
}

demo_prune_fpgm_v2_T (){
cd ${slim_dir}/demo/prune
python train.py \
    --model="MobileNetV2" \
    --pretrained_model="../pretrain/MobileNetV2_pretrained" \
    --data="imagenet" \
    --pruned_ratio=0.325 \
    --lr=0.001 \
    --num_epochs=2 \
    --test_period=1 \
    --step_epochs 30 60 80 \
    --l2_decay=1e-4 \
    --lr_strategy="piecewise_decay" \
    --criterion="geometry_median" \
    --model_path="./output/fpgm_mobilenetv2_models" \
    --save_inference True --use_gpu False >${log_path}/slim_prune_fpgm_v2_T 2>&1
print_info $? slim_prune_fpgm_v2_T
python eval.py --model "MobileNetV2" --data "imagenet" \
--model_path "./output/fpgm_mobilenetv2_models/0" --use_gpu False >${log_path}/slim_prune_fpgm_v2_eval 2>&1
print_info $? slim_prune_fpgm_v2_eval
}

# dygraph_prune
demo_dy_prune_ResNet34_f42(){
cd ${slim_dir}/demo/dygraph/pruning || catchException demo_dy_prune_ResNet34_f42
ln -s ${slim_dir}/demo/data data
python train.py \
    --use_gpu=False \
    --model="resnet34" \
    --data="imagenet" \
    --pruned_ratio=0.25 \
    --num_epochs=1 \
    --batch_size=128 \
    --lr_strategy="cosine_decay" \
    --criterion="fpgm" \
    --model_path="./fpgm_resnet34_025_120_models" --use_gpu False >${log_path}/dy_prune_ResNet34_f42 2>&1
print_info $? dy_prune_ResNet34_f42

#2.3 恢复训练  通过设置checkpoint选项进行恢复训练：
python train.py \
    --use_gpu=False \
    --model="resnet34" \
    --data="imagenet" \
    --pruned_ratio=0.25 \
    --num_epochs=2 \
    --batch_size=128 \
    --lr_strategy="cosine_decay" \
    --criterion="fpgm" \
    --model_path="./fpgm_resnet34_025_120_models" \
    --checkpoint="./fpgm_resnet34_025_120_models/0" --use_gpu False >${log_path}/dy_prune_ResNet50_f42_load 2>&1
print_info $? dy_prune_ResNet50_f42_load

}

# 3unstructured_prune
demo_st_unstructured_prune(){
cd ${slim_dir}/demo/unstructured_prune || catchException demo_st_unstructured_prune

python train.py \
--batch_size 256 \
--pretrained_model ../pretrain/MobileNetV1_pretrained \
--lr 0.05 \
--pruning_mode ratio \
--ratio 0.55 \
--data imagenet \
--lr_strategy piecewise_decay \
--step_epochs 1 2 3 \
--num_epochs 1 \
--test_period 1 \
--model_period 1 \
--model_path st_ratio_models \
--use_gpu False >${log_path}/st_unstructured_prune_ratio_T 2>&1
print_info $? st_unstructured_prune_ratio_T
}

demo_dy_unstructured_prune(){
# dy_threshold
cd ${slim_dir}/demo/dygraph/unstructured_pruning || catchException demo_dy_unstructured_prune
python  train.py \
--data imagenet \
--lr 0.05 \
--pruning_mode threshold \
--threshold 0.01 \
--batch_size 256 \
--lr_strategy piecewise_decay \
--step_epochs 1 2 3 \
--num_epochs 1 \
--test_period 1 \
--model_period 1 \
--model_path dy_threshold_models --use_gpu False >${log_path}/dy_threshold_prune_T 2>&1
print_info $? dy_threshold_prune_T

}

all_prune(){
  demo_prune_v1
  demo_prune_fpgm_v2_T
  demo_dy_prune_ResNet34_f42
  demo_st_unstructured_prune
  demo_dy_unstructured_prune
}

#4 nas
demo_nas(){
cd ${slim_dir}/demo/nas || catchException demo_nas
python sa_nas_mobilenetv2.py --search_steps 1 --port 8881 --retain_epoch 1 --use_gpu False >${log_path}/sa_nas_v2_T_1card 2>&1
print_info $? sa_nas_v2_T_1card
}
all_nas(){ 
    demo_nas
}

export all_case_list=(all_distillation all_quant all_prune  all_nas)

####################################
echo --- start run case ---
case_num=1
for model in ${all_case_list[*]};do
    echo ---$case_num/${#all_case_list[*]}: ${model}---
    ${model}
    let case_num++
done
echo --- end run case---
