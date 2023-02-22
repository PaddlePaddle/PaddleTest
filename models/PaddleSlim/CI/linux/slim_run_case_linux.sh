#!/usr/bin/env bash
# run_CI/run_CE/ALL 、cudaid1、cudaid2
echo ---run slim case with $1 ---

print_info(){
if [ $1 -ne 0 ];then
    mv ${log_path}/$2 ${log_path}/FAIL_$2.log
    echo ---${log_path}/FAIL_$2---
    echo "fail log as follow"
    cat  ${log_path}/FAIL_$2.log
else
    mv ${log_path}/$2 ${log_path}/SUCCESS_$2.log
    echo ---${log_path}/SUCCESS_$2---
    cat  ${log_path}/SUCCESS_$2.log | grep -i 'Memory Usage'
fi
}

catchException() {
  echo $1 failed due to exception >> FAIL_Exception.log
}

cudaid1=$2;
cudaid2=$3;
echo ---cudaid1:${cudaid1}---
echo ---cudaid2:${cudaid2}---

export CUDA_VISIBLE_DEVICES=${cudaid2}
# 1 dist
demo_distill_01(){
cd ${slim_dir}/demo/distillation || catchException demo_distill_01
if [ -d "output" ];then
    rm -rf output
fi
CUDA_VISIBLE_DEVICES=${cudaid1}
python distill.py --num_epochs 1 --batch_size 64 --save_inference True > ${log_path}/st_distill_ResNet50_vd_MobileNet 2>&1
print_info $? st_distill_ResNet50_vd_MobileNet
}

demo_distill_02(){
CUDA_VISIBLE_DEVICES=${cudaid1}
cd ${slim_dir}/demo/distillation || catchException demo_distill_02
if [ -d "output" ];then
    rm -rf output
fi

python distill.py --num_epochs 1 --batch_size 64 --save_inference True \
--model ResNet50 --teacher_model ResNet101_vd \
--teacher_pretrained_model ../pretrain/ResNet101_vd_pretrained > ${log_path}/st_distill_ResNet101_vd_ResNet50 2>&1
print_info $? st_distill_ResNet101_vd_ResNet50
}

demo_distill_03(){
CUDA_VISIBLE_DEVICES=${cudaid1}
cd ${slim_dir}/demo/distillation || catchException demo_distill_03
if [ -d "output" ];then
    rm -rf output
fi

python distill.py --num_epochs 1 --batch_size 64 --save_inference True \
--model MobileNetV2_x0_25 --teacher_model MobileNetV2 \
--teacher_pretrained_model ../pretrain/MobileNetV2_pretrained >${log_path}/st_distill_MobileNetV2_MobileNetV2_x0_25 2>&1
print_info $? st_distill_MobileNetV2_MobileNetV2_x0_25
}

demo_deep_mutual_learning_01(){
cd ${slim_dir}/demo/deep_mutual_learning || catchException st_dml_mv1_mv1
python dml_train.py --epochs 1 >${log_path}/dml_mv1_mv1 2>&1
print_info $? st_dml_mv1_mv1

}

demo_deep_mutual_learning_02(){
cd ${slim_dir}/demo/deep_mutual_learning || catchException demo_deep_mutual_learning_02
python dml_train.py --models='mobilenet-resnet50' --batch_size 128 --epochs 1 >${log_path}/st_dml_mv1_res50 2>&1
print_info $? st_dml_mv1_res50
}

# distill + dml 共计5个case
all_distill_CI(){
    demo_distill_01
}

all_distill_CE(){
    demo_distill_02
    demo_distill_03
    #demo_deep_mutual_learning_01
    #demo_deep_mutual_learning_02
}

all_distill_ALL(){
    demo_distill_01
    demo_distill_02
    demo_distill_03
    #demo_deep_mutual_learning_01
    #demo_deep_mutual_learning_02
}

demo_st_quant_aware_v1(){
CUDA_VISIBLE_DEVICES=${cudaid1}

cd ${slim_dir}/demo/quant/quant_aware || catchException demo_st_quant_aware_v1
if [ -d "output" ];then
    rm -rf output
fi

python train.py --model MobileNet --pretrained_model ../../pretrain/MobileNetV1_pretrained \
--checkpoint_dir ./output/mobilenetv1 --num_epochs 1 --batch_size 16 >${log_path}/st_quant_aware_v1 2>&1
print_info $? st_quant_aware_v1
}

demo_st_quant_aware_ResNet34(){
CUDA_VISIBLE_DEVICES=${cudaid1}
cd ${slim_dir}/demo/quant/quant_aware || catchException demo_st_quant_aware_ResNet34
if [ -d "output" ];then
    rm -rf output
fi

python train.py --model ResNet34 \
--pretrained_model ../../pretrain/ResNet34_pretrained \
--checkpoint_dir ./output/ResNet34 --num_epochs 1 >${log_path}/st_quant_aware_ResNet34 2>&1
print_info $? st_quant_aware_ResNet34
}

demo_st_quant_embedding(){
CUDA_VISIBLE_DEVICES=${cudaid1}
cd ${slim_dir}/demo/quant/quant_embedding || catchException demo_st_quant_embedding

if [ -d "data" ];then
    rm -rf data
fi
wget -q https://sys-p0.bj.bcebos.com/slim_ci/word_2evc_demo_data.tar.gz --no-check-certificate
tar xf word_2evc_demo_data.tar.gz
mv word_2evc_demo_data data
if [ -d "v1_cpu5_b100_lr1dir" ];then
    rm -rf v1_cpu5_b100_lr1dir
fi

OPENBLAS_NUM_THREADS=1 CPU_NUM=5 python train.py --train_data_dir data/convert_text8 \
--dict_path data/test_build_dict --num_passes 1 --batch_size 100 --model_output_dir v1_cpu5_b100_lr1dir \
 --base_lr 1.0 --print_batch 1000 --with_speed --is_sparse >${log_path}/st_quant_em_word2vec 2>&1
print_info $? st_quant_em_word2vec

python infer.py --infer_epoch --test_dir data/test_mid_dir \
--dict_path data/test_build_dict_word_to_id_ \
--batch_size 20000 --model_dir v1_cpu5_b100_lr1dir/  --start_index 0 \
--last_index 0 --emb_quant True >${log_path}/st_quant_em_after_infer 2>&1
print_info $? st_quant_em_after_infer
}

demo_st_quant_post_hist(){
CUDA_VISIBLE_DEVICES=${cudaid1}
cd ${slim_dir}/demo/quant/quant_post || catchException demo_st_quant_post_hist

wget -qP inference_model https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV1_infer.tar
cd inference_model/
tar -xf MobileNetV1_infer.tar
cd ..

for algo in hist
do
# 带bc参数、指定algo、且为新存储格式的离线量化
python quant_post.py --model_path ./inference_model/MobileNetV1_infer/ \
--save_path ./quant_model/${algo}_bc/MobileNetV1 \
--algo ${algo} --bias_correction True \
--onnx_format=True  --ce_test=True  --is_full_quantize=True >${log_path}/st_quant_post_T_${algo}_bc_onnx_format 2>&1
print_info $? st_quant_post_T_${algo}_bc_onnx_format

# 量化后eval
echo "quant_post eval bc " ${algo}
python eval.py --model_path ./quant_model/${algo}_bc/MobileNetV1 > ${log_path}/st_quant_post_${algo}_bc_onnx_format_eval 2>&1
print_info $? st_quant_post_${algo}_bc_onnx_format_eval

#old 存储格式
python quant_post.py --model_path ./inference_model/MobileNetV1_infer/ \
--save_path ./quant_model/${algo}_bc_adaround/MobileNetV1 \
--algo ${algo} --bias_correction True \
--round_type=adaround >${log_path}/st_quant_post_T_${algo}_bc_old_format 2>&1
print_info $? st_quant_post_T_${algo}_bc_old_format

python eval.py --model_path ./quant_model/${algo}_bc_adaround/MobileNetV1 > ${log_path}/st_quant_post_${algo}_bc_old_format_eval 2>&1
print_info $? st_quant_post_${algo}_bc_old_format_eval

done
}

demo_st_quant_post(){
CUDA_VISIBLE_DEVICES=${cudaid1}
cd ${slim_dir}/demo/quant/quant_post || catchException demo_st_quant_post

wget -qP inference_model https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV1_infer.tar
cd inference_model/
tar -xf MobileNetV1_infer.tar
cd ..

for algo in avg mse emd
do
# 带bc参数、指定algo、且为新存储格式的离线量化
python quant_post.py --model_path ./inference_model/MobileNetV1_infer/ \
--save_path ./quant_model/${algo}_bc/MobileNetV1 \
--algo ${algo} --bias_correction True >${log_path}/st_quant_post_T_${algo}_bc 2>&1
print_info $? st_quant_post_T_${algo}_bc

# 量化后eval
echo "quant_post eval bc " ${algo}
python eval.py --model_path ./quant_model/${algo}_bc/MobileNetV1  > ${log_path}/st_quant_post_${algo}_bc_eval 2>&1
print_info $? st_quant_post_${algo}_bc_eval
done
}

demo_st_quant_post_hpo(){
CUDA_VISIBLE_DEVICES=${cudaid1}
cd ${slim_dir}/demo/quant/quant_post_hpo || catchException demo_st_quant_post_hpo

wget -qP inference_model https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV1_infer.tar
cd inference_model/
tar -xf MobileNetV1_infer.tar
cd ..

# 2. quant_post_hpo 设置max_model_quant_count=2
python quant_post_hpo.py  \
--use_gpu=True     \
--model_path=./inference_model/MobileNetV1_infer/   \
--save_path=./inference_model/MobileNetV1_quant/  \
--model_filename="inference.pdmodel" \
--params_filename="inference.pdiparams" \
--max_model_quant_count=1 > ${log_path}/st_quant_post_hpo 2>&1
print_info $? st_quant_post_hpo
# 3. 量化后eval
# python ../quant_post/eval.py \
# --model_path=./inference_model/MobileNetV1_quant/ > ${log_path}/st_quant_post_hpo_eval 2>&1
# print_info $? st_quant_post_hpo_eval
}

demo_st_pact_quant_aware_v3(){
CUDA_VISIBLE_DEVICES=${cudaid1}

cd ${slim_dir}/demo/quant/pact_quant_aware || catchException demo_st_pact_quant_aware_v3

python train.py --model MobileNetV3_large_x1_0 \
--pretrained_model ../../pretrain/MobileNetV3_large_x1_0_ssld_pretrained \
--num_epochs 1 --lr 0.0001 --use_pact False --batch_size 32 >${log_path}/st_pact_quant_aware_v3_no_pact 2>&1
print_info $? st_pact_quant_aware_v3_no_pact

python train.py --model MobileNetV3_large_x1_0 \
--pretrained_model ../../pretrain/MobileNetV3_large_x1_0_ssld_pretrained \
--num_epochs 1 --lr 0.0001 --use_pact True --batch_size 64 --lr_strategy=piecewise_decay \
--step_epochs 2 --l2_decay 1e-5 >${log_path}/st_pact_quant_aware_v3_with_pact 2>&1
print_info $? st_pact_quant_aware_v3_with_pact
# load
python train.py --model MobileNetV3_large_x1_0 \
--pretrained_model ../../pretrain/MobileNetV3_large_x1_0_ssld_pretrained \
--num_epochs 2 --lr 0.0001 --use_pact True --batch_size 64 --lr_strategy=piecewise_decay \
--step_epochs 20 --l2_decay 1e-5 \
--checkpoint_dir ./output/MobileNetV3_large_x1_0/0 \
--checkpoint_epoch 0 >${log_path}/st_pact_quant_aware_v3_load 2>&1
print_info $? st_pact_quant_aware_v3_load
}

#cd demo/quant/quant_aware_with_infermodel/ 所需训练时间较长，UT中自定义model覆盖

all_st_quant_CI(){
    #demo_st_quant_aware_v1
    demo_st_quant_aware_ResNet34
    demo_st_quant_post_hist
    demo_st_pact_quant_aware_v3
}

all_st_quant_CE(){
    demo_st_quant_post_hpo
    demo_st_quant_embedding
    demo_st_quant_post
}

all_st_quant_ALL(){
    demo_st_quant_post_hpo
    demo_st_quant_post_hist
    demo_st_quant_aware_v1
    demo_st_pact_quant_aware_v3
    #demo_st_quant_aware_ResNet34
    demo_st_quant_embedding
    demo_st_quant_post
}

demo_dy_quant_v1(){
cd ${slim_dir}/demo/dygraph/quant || catchException demo_dy_quant_v1
python train.py --model='mobilenet_v1' \
--pretrained_model '../../pretrain/MobileNetV1_pretrained' \
--num_epochs 1 \
--batch_size 128 > ${log_path}/dy_quant_v1 2>&1
print_info $? dy_quant_v1
}

demo_dy_quant_v3(){
cd ${slim_dir}/demo/dygraph/quant || catchException demo_dy_quant_v3
python train.py  --lr=0.001 \
--batch_size 128 \
--use_pact=True --num_epochs=1 --l2_decay=2e-5 --ls_epsilon=0.1 \
--pretrained_model ../../pretrain/MobileNetV3_large_x1_0_ssld_pretrained \
--num_epochs 1 > ${log_path}/dy_pact_quant_v3_gpu1 2>&1
print_info $? dy_pact_quant_v3_gpu1
# 多卡训练
CUDA_VISIBLE_DEVICES=${cudaid2}
python -m paddle.distributed.launch \
train.py  --lr=0.001 \
--pretrained_model ../../pretrain/MobileNetV3_large_x1_0_ssld_pretrained \
--use_pact=True --num_epochs=1 \
--l2_decay=2e-5 \
--ls_epsilon=0.1 \
--batch_size=128 \
--model_save_dir output > ${log_path}/dy_pact_quant_v3_gpu2 2>&1
print_info $? dy_pact_quant_v3_gpu2
}

ce_tests_dy_qat4(){
cd ${slim_dir}/ce_tests/dygraph/quant || catchException ce_tests_dy_qat4

ln -s ${slim_dir}/demo/data/ILSVRC2012
# if set as -1, use all test samples
test_samples=1000
data_path='./ILSVRC2012/'
epoch=1
lr=0.0001
batch_size=32
num_workers=3
output_dir=$PWD/output_qat


for model in mobilenet_v1 mobilenet_v2 resnet50 vgg16
do
    echo "---qat quant model: ${model}---"
    python ./src/qat.py \
        --arch=${model} \
        --data=${data_path} \
        --epoch=${epoch} \
        --batch_size=${batch_size} \
        --num_workers=${num_workers} \
        --lr=${lr} \
        --output_dir=${output_dir} \
        --enable_quant > ${log_path}/dy_qat_${model} 2>&1
    print_info $? dy_qat_${model}

    echo "---save qat int8: ${model}---"
    python src/save_quant_model.py \
    --load_model_path output_qat/${model}  \
    --save_model_path int8_qat_models/${model}  > ${log_path}/dy_qat_save_${model} 2>&1
    print_info $? dy_qat_save_${model}

    echo "---eval qat fp32_infer : ${model}---"
    python ./src/test.py \
   	--model_path=output_qat/${model}  \
   	--data_dir=${data_path} \
   	--test_samples=-1 \
   	--batch_size=32 \
   	--use_gpu=True \
   	--ir_optim=False > ${log_path}/dy_qat_eval_fp32_${model} 2>&1
    print_info $? dy_qat_eval_fp32_${model}

#    从2022.10.20开始不再支持
#    echo "---eval qat int8_infer : ${model}---"
#    python ./src/test.py \
#   	--model_path=int8_qat_models/${model}  \
#   	--data_dir=${data_path} \
#   	--test_samples=-1 \
#   	--batch_size=32 \
#   	--use_gpu=False \
#   	--ir_optim=False  > ${log_path}/dy_qat_eval_int8_${model} 2>&1
#    print_info $? dy_qat_eval_int8_${model}
done

}

ce_tests_dy_ptq4(){
cd ${slim_dir}/ce_tests/dygraph/quant || catchException ce_tests_dygraph_ptq4
ln -s ${slim_dir}/demo/data/ILSVRC2012
test_samples=1000  # if set as -1, use all test samples
data_path='./ILSVRC2012/'
batch_size=32
epoch=1
output_dir="./output_ptq"
quant_batch_num=10
quant_batch_size=10
for model in mobilenet_v1 mobilenet_v2 resnet50 vgg16
do
    echo "--------quantize model: ${model}-------------"
    export CUDA_VISIBLE_DEVICES=${cudaid1}
    # save ptq quant model
    python ./src/ptq.py \
        --data=${data_path} \
        --arch=${model} \
        --quant_batch_num=${quant_batch_num} \
        --quant_batch_size=${quant_batch_size} \
        --output_dir=${output_dir} > ${log_path}/dy_ptq_${model} 2>&1
        print_info $? dy_ptq_${model}

    echo "-------- eval fp32_infer model -------------", ${model}
    python ./src/test.py \
        --model_path=${output_dir}/${model}/fp32_infer \
        --data_dir=${data_path} \
        --batch_size=${batch_size} \
        --use_gpu=True \
        --test_samples=${test_samples} \
        --ir_optim=False > ${log_path}/dy_ptq_eval_fp32_${model} 2>&1
        print_info $? dy_ptq_eval_fp32_${model}

#    echo "-------- eval int8_infer model -------------", ${model}
#    python ./src/test.py \
#        --model_path=${output_dir}/${model}/int8_infer \
#        --data_dir=${data_path} \
#        --batch_size=${batch_size} \
#        --use_gpu=False \
#        --test_samples=${test_samples} \
#        --ir_optim=False > ${log_path}/dy_ptq_eval_int8_${model} 2>&1
#        print_info $? dy_ptq_eval_int8_${model}
done
}

demo_ptq_yolo_series(){
cd ${slim_dir}/example/post_training_quantization/pytorch_yolo_series/

wget -q https://paddle-slim-models.bj.bcebos.com/act/yolov6s.onnx
wget https://paddle-qa.bj.bcebos.com/PaddleDetection/coco.zip
unzip -q coco.zip

sed -i 's/epochs=20/epochs=1/' fine_tune.py
sed -i 's/\/dataset\/coco/coco/' ./configs/yolov6s_fine_tune.yaml

python fine_tune.py \
--config_path=./configs/yolov6s_fine_tune.yaml \
--recon_level=region-wise \
--save_dir=region_ptq_out > ${log_path}/ptq_yolo_series_region_train 2>&1
print_info $? ptq_yolo_series_region_train

sed -i  's/yolov6s.onnx/region_ptq_out/' ./configs/yolov6s_fine_tune.yaml
python eval.py \
--config_path=./configs/yolov6s_fine_tune.yaml > ${log_path}/ptq_yolo_series_region_eval 2>&1
print_info $? ptq_yolo_series_region_eval

sed -i  's/region_ptq_out/yolov6s.onnx/' ./configs/yolov6s_fine_tune.yaml
python fine_tune.py \
--config_path=./configs/yolov6s_fine_tune.yaml \
--recon_level=layer-wise \
--save_dir=layer_ptq_out > ${log_path}/ptq_yolo_series_layer_train 2>&1
print_info $? ptq_yolo_series_layer_train

sed -i  's/yolov6s.onnx/layer_ptq_out/' ./configs/yolov6s_fine_tune.yaml
python eval.py \
--config_path=./configs/yolov6s_fine_tune.yaml > ${log_path}/ptq_yolo_series_layer_eval 2>&1
print_info $? ptq_yolo_series_layer_eval
}

all_dy_quant_CI(){
    demo_dy_quant_v1
}

all_dy_quant_CE(){
    demo_dy_quant_v3
    ce_tests_dy_qat4
    ce_tests_dy_ptq4
    demo_ptq_yolo_series
}

all_dy_quant_ALL(){
    demo_dy_quant_v1
    demo_dy_quant_v3
    ce_tests_dy_qat4
    ce_tests_dy_ptq4
    demo_ptq_yolo_series
}

demo_st_prune_v1(){
cd ${slim_dir}/demo/prune  || catchException demo_st_prune_v1

if [ -d "models" ];then
    rm -rf models
fi

python train.py --model "MobileNet" --pruned_ratio 0.31 --data "imagenet" \
--pretrained_model ../pretrain/MobileNetV1_pretrained/ --num_epochs 1 >${log_path}/st_prune_v1_T 2>&1
print_info $? st_prune_v1_T
}

demo_st_prune_fpgm_v1(){
cd ${slim_dir}/demo/prune  || catchException demo_st_prune_fpgm_v1
if [ -d "models" ];then
    rm -rf models
fi
python train.py \
    --model="MobileNet" \
    --pretrained_model="../pretrain/MobileNetV1_pretrained" \
    --data="imagenet" \
    --pruned_ratio=0.3125 \
    --lr=0.1 \
    --num_epochs=1 \
    --test_period=1 \
    --step_epochs 30 60 90\
    --l2_decay=3e-5 \
    --lr_strategy="piecewise_decay" \
    --criterion="geometry_median" \
    --model_path="./fpgm_mobilenetv1_models" \
    --save_inference True  >${log_path}/st_prune_fpgm_v1 2>&1
print_info $? st_prune_fpgm_v1
}

demo_st_prune_fpgm_v2(){
cd ${slim_dir}/demo/prune  || catchException demo_st_prune_fpgm_v2
if [ -d "models" ];then
    rm -rf models
fi
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
    --save_inference True >${log_path}/st_prune_fpgm_v2 2>&1
print_info $? st_prune_fpgm_v2
python eval.py --model "MobileNetV2" --data "imagenet" \
--model_path "./output/fpgm_mobilenetv2_models/0" >${log_path}/st_prune_fpgm_v2_eval 2>&1
print_info $? st_prune_fpgm_v2_eval
}

demo_st_prune_fpgm_resnet34_50(){
cd ${slim_dir}/demo/prune  || catchException demo_st_prune_fpgm_resnet34_50
if [ -d "models" ];then
    rm -rf models
fi
python train.py \
    --model="ResNet34" \
    --pretrained_model="../pretrain/ResNet34_pretrained" \
    --data="imagenet" \
    --pruned_ratio=0.3125 \
    --lr=0.001 \
    --num_epochs=2 \
    --test_period=1 \
    --step_epochs 30 60 \
    --l2_decay=1e-4 \
    --lr_strategy="piecewise_decay" \
    --criterion="geometry_median" \
    --model_path="./output/fpgm_resnet34_50_models" \
    --save_inference True >${log_path}/st_prune_fpgm_resnet34_50 2>&1
print_info $? st_prune_fpgm_resnet34_50
python eval.py --model "ResNet34" --data "imagenet" \
--model_path "./output/fpgm_resnet34_50_models/0" >${log_path}/st_prune_fpgm_resnet34_50_eval 2>&1
print_info $? st_prune_fpgm_resnet34_50_eval
}

demo_st_prune_ResNet50(){
cd ${slim_dir}/demo/prune  || catchException demo_st_prune_ResNet50
if [ -d "models" ];then
    rm -rf models
fi
python train.py --model ResNet50 --pruned_ratio 0.31 --data "imagenet" \
--save_inference True --pretrained_model ../pretrain/ResNet50_pretrained \
--num_epochs 1 --batch_size 128 >${log_path}/st_prune_ResNet50 2>&1
print_info $? st_prune_ResNet50
}

all_st_prune_CI(){
    demo_st_prune_v1
    demo_st_prune_fpgm_v2
}

all_st_prune_CE(){
    demo_st_prune_fpgm_v1
    demo_st_prune_fpgm_resnet34_50
    demo_st_prune_ResNet50

}

all_st_prune_ALL(){
    demo_st_prune_v1
    demo_st_prune_fpgm_v2
    demo_st_prune_fpgm_v1
    demo_st_prune_fpgm_resnet34_50
    demo_st_prune_ResNet50
}

demo_dy_pruning_resnet34(){
cd ${slim_dir}/demo/dygraph/pruning  || catchException demo_dy_pruning_resnet34
ln -s ${slim_dir}/demo/data data
python train.py \
    --use_gpu=True \
    --model="resnet34" \
    --data="imagenet" \
    --pruned_ratio=0.25 \
    --num_epochs=1 \
    --batch_size=128 \
    --lr_strategy="cosine_decay" \
    --criterion="fpgm" \
    --model_path="./fpgm_resnet34_025_120_models" >${log_path}/dy_prune_ResNet34_f42_gpu1 2>&1
print_info $? dy_prune_ResNet34_f42_gpu1
# 恢复训练  通过设置checkpoint选项进行恢复训练：
python train.py \
    --use_gpu=True \
    --model="resnet34" \
    --data="imagenet" \
    --pruned_ratio=0.25 \
    --num_epochs=2 \
    --batch_size=128 \
    --lr_strategy="cosine_decay" \
    --criterion="fpgm" \
    --model_path="./fpgm_resnet34_025_120_models" \
    --checkpoint="./fpgm_resnet34_025_120_models/0" >${log_path}/dy_prune_ResNet34_f42_gpu1_load 2>&1
print_info $? dy_prune_ResNet34_f42_gpu1_load

python eval.py \
--checkpoint=./fpgm_resnet34_025_120_models/1 \
--model="resnet34" \
--pruned_ratio=0.25 \
--batch_size=128 >${log_path}/dy_prune_ResNet34_f42_gpu1_eval 2>&1
print_info $? dy_prune_ResNet34_f42_gpu1_eval

#导出模型
CUDA_VISIBLE_DEVICES=${cudaid1} python export_model.py \
--checkpoint=./fpgm_resnet34_025_120_models/final \
--model="resnet34" \
--pruned_ratio=0.25 \
--output_path=./infer_final/resnet > ${log_path}/dy_prune_ResNet34_f42_gpu1_export 2>&1
print_info $? dy_prune_ResNet34_f42_gpu1_export
}

demo_dy_pruning_v1(){
cd ${slim_dir}/demo/dygraph/pruning  || catchException demo_dy_pruning_v1
ln -s ${slim_dir}/demo/data data

CUDA_VISIBLE_DEVICES=${cudaid2}
python -m paddle.distributed.launch \
--log_dir="fpgm_mobilenetv1_train_log" \
train.py \
    --model="mobilenet_v1" \
    --data="imagenet" \
    --pruned_ratio=0.3125 \
    --lr=0.1 \
    --num_epochs=1 \
    --test_period=1 \
    --step_epochs 30 60 90\
    --l2_decay=3e-5 \
    --lr_strategy="piecewise_decay" \
    --criterion="fpgm" \
    --model_path="./fpgm_mobilenetv1_models" > ${log_path}/dy_prune_fpgm_mobilenetv1_50_T 2>&1
print_info $? dy_prune_fpgm_mobilenetv1_50_T
}

demo_dy_pruning_v2(){
cd ${slim_dir}/demo/dygraph/pruning  || catchException demo_dy_pruning_resnet34
ln -s ${slim_dir}/demo/data data

CUDA_VISIBLE_DEVICES=${cudaid2}
python -m paddle.distributed.launch \
--log_dir="fpgm_mobilenetv2_train_log" \
train.py \
    --model="mobilenet_v2" \
    --data="imagenet" \
    --pruned_ratio=0.325 \
    --lr=0.001 \
    --num_epochs=1 \
    --test_period=1 \
    --step_epochs 30 60 80\
    --l2_decay=1e-4 \
    --lr_strategy="piecewise_decay" \
    --criterion="fpgm" \
    --model_path="./fpgm_mobilenetv2_models" > ${log_path}/dy_prune_fpgm_mobilenetv2_50_T 2>&1
print_info $? dy_prune_fpgm_mobilenetv2_50_T
}

demo_dy_pruning_ResNet34_f42(){
cd ${slim_dir}/demo/dygraph/pruning  || catchException demo_dy_pruning_resnet34
ln -s ${slim_dir}/demo/data data

CUDA_VISIBLE_DEVICES=${cudaid2}
python -m paddle.distributed.launch \
--log_dir="fpgm_resnet34_f_42_train_log" \
train.py \
    --use_gpu=True \
    --model="resnet34" \
    --data="imagenet" \
    --pruned_ratio=0.25 \
    --batch_size=128 \
    --num_epochs=1 \
    --test_period=1 \
    --lr_strategy="cosine_decay" \
    --criterion="fpgm" \
    --model_path="./fpgm_resnet34_025_120_models" > ${log_path}/dy_prune_ResNet34_f42_gpu2 2>&1
print_info $? dy_prune_ResNet34_f42_gpu2
}

all_dy_prune_CI(){
    demo_dy_pruning_resnet34
}

all_dy_prune_CE(){
	demo_dy_pruning_v1
	demo_dy_pruning_v2
  demo_dy_pruning_ResNet34_f42
}


all_dy_prune_ALL(){
    demo_dy_pruning_resnet34
    demo_dy_pruning_v1
    demo_dy_pruning_v2
    demo_dy_pruning_ResNet34_f42
}

demo_st_unstructured_prune_threshold(){
cd ${slim_dir}/demo/unstructured_prune  || catchException demo_st_unstructured_prune_threshold
python train.py \
--batch_size 256 \
--pretrained_model ../pretrain/MobileNetV1_pretrained \
--lr 0.05 \
--pruning_mode threshold \
--threshold 0.01 \
--data imagenet \
--lr_strategy piecewise_decay \
--step_epochs 1 2 3 \
--num_epochs 1 \
--test_period 1 \
--model_period 1 \
--model_path st_unstructured_models > ${log_path}/st_unstructured_prune_threshold_T 2>&1
print_info $? st_unstructured_prune_threshold_T
# eval
python evaluate.py \
       --pruned_model=st_unstructured_models \
       --data="imagenet"  >${log_path}/st_unstructured_prune_threshold_eval 2>&1
print_info $? st_unstructured_prune_threshold_eval
}

demo_st_unstructured_prune_ratio(){
cd ${slim_dir}/demo/unstructured_prune  || catchException demo_st_unstructured_prune_ratio
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
--model_path st_ratio_models >${log_path}/st_unstructured_prune_ratio_T 2>&1
print_info $? st_unstructured_prune_ratio_T
}

demo_st_unstructured_prune_ratio_gmp(){
cd ${slim_dir}/demo/unstructured_prune  || catchException demo_st_unstructured_prune_ratio_gmp
export CUDA_VISIBLE_DEVICES=${cudaid2}
python -m paddle.distributed.launch \
          --log_dir="st_unstructured_prune_gmp_log" \
          train.py \
          --batch_size 64 \
          --data imagenet \
          --pruning_mode ratio \
          --ratio 0.75 \
          --lr 0.005 \
          --model MobileNet \
          --num_epochs 1 \
          --test_period 5 \
          --model_period 10 \
          --pretrained_model ../pretrain/MobileNetV1_pretrained \
          --model_path "./models" \
          --step_epochs  71 88 \
          --initial_ratio 0.15 \
          --pruning_steps 5 \
          --stable_epochs 0 \
          --pruning_epochs 54 \
          --tunning_epochs 54 \
          --last_epoch -1 \
          --prune_params_type conv1x1_only \
          --pruning_strategy gmp > ${log_path}/st_unstructured_prune_ratio_gmp 2>&1
print_info $? st_unstructured_prune_ratio_gmp
}

all_st_unstr_prune_CI(){
    demo_st_unstructured_prune_threshold
    demo_st_unstructured_prune_ratio_gmp
}

all_st_unstr_prune_CE(){
    demo_st_unstructured_prune_ratio

}

all_st_unstr_prune_ALL(){
    demo_st_unstructured_prune_threshold
    demo_st_unstructured_prune_ratio
    demo_st_unstructured_prune_ratio_gmp
}

demo_dy_unstructured_pruning_ratio(){
cd ${slim_dir}/demo/dygraph/unstructured_pruning || catchException demo_dy_unstructured_pruning_ratio
export CUDA_VISIBLE_DEVICES=${cudaid2}
python -m paddle.distributed.launch \
--log_dir train_dy_ratio_log train.py \
--data imagenet \
--lr 0.05 \
--pruning_mode ratio \
--ratio 0.55 \
--batch_size 256 \
--lr_strategy piecewise_decay \
--step_epochs 1 2 3 \
--num_epochs 1 \
--test_period 1 \
--model_period 1 \
--model_path dy_ratio_models >${log_path}/dy_unstructured_prune_ratio_T 2>&1
print_info $? dy_unstructured_prune_ratio_T
}

demo_dy_unstructured_pruning_threshold(){
cd ${slim_dir}/demo/dygraph/unstructured_pruning || catchException demo_dy_unstructured_pruning_threshold
export CUDA_VISIBLE_DEVICES=${cudaid2}
python -m paddle.distributed.launch \
--log_dir train_dy_threshold_log train.py \
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
--model_path dy_threshold_models >${log_path}/dy_threshold_prune_T 2>&1
print_info $? dy_threshold_prune_T
# eval
python evaluate.py --pruned_model dy_threshold_models/model.pdparams \
--data imagenet >${log_path}/dy_threshold_prune_eval 2>&1
print_info $? dy_threshold_prune_eval

# load
python -m paddle.distributed.launch \
--log_dir train_dy_threshold_load_log train.py \
--data imagenet \
--lr 0.05 \
--pruning_mode threshold \
--threshold 0.01 \
--batch_size 256 \
--lr_strategy piecewise_decay \
--step_epochs 1 2 3 \
--num_epochs 3 \
--test_period 1 \
--model_period 1 \
--model_path dy_threshold_models_new \
--pretrained_model dy_threshold_models/model.pdparams \
--last_epoch 1 > ${log_path}/dy_threshold_prune_T_load 2>&1
print_info $? dy_threshold_prune_T_load
}

demo_dy_unstructured_pruning_ratio_gmp(){
cd ${slim_dir}/demo/dygraph/unstructured_pruning || catchException demo_dy_unstructured_pruning_ratio
export CUDA_VISIBLE_DEVICES=${cudaid2}
python -m paddle.distributed.launch \
          --log_dir="dy_unstructured_prune_gmp_log" \
          train.py \
          --batch_size 128 \
          --data imagenet \
          --pruning_mode ratio \
          --ratio 0.75 \
          --lr 0.005 \
          --num_epochs 1 \
          --test_period 5 \
          --model_period 10 \
          --model_path "./models" \
          --step_epochs 71 88 \
          --initial_ratio 0.15 \
          --pruning_steps 2 \
          --stable_epochs 0 \
          --pruning_epochs 100 \
          --tunning_epochs 54 \
          --last_epoch -1 \
          --pruning_strategy gmp \
          --prune_params_type conv1x1_only > ${log_path}/dy_unstructured_prune_ratio_gmp 2>&1
print_info $? dy_unstructured_prune_ratio_gmp
}

all_dy_unstr_prune_CI(){
    demo_dy_unstructured_pruning_threshold
    #demo_dy_unstructured_pruning_ratio_gmp
}

all_dy_unstr_prune_CE(){
    demo_dy_unstructured_pruning_ratio
}

all_dy_unstr_prune_ALL(){
    demo_dy_unstructured_pruning_ratio
    demo_dy_unstructured_pruning_threshold
    demo_dy_unstructured_pruning_ratio_gmp
}

demo_sa_nas(){
cd ${slim_dir}/demo/nas  || catchException demo_nas
model=demo_nas_sa_nas_v2_T_1card
CUDA_VISIBLE_DEVICES=${cudaid1} python sa_nas_mobilenetv2.py --search_steps 1  --retain_epoch 1 --port 8881 >${log_path}/${model} 2>&1
print_info $? ${model}
}

demo_nas4(){
cd ${slim_dir}/demo/nas || catchException demo_nas4
model=sa_nas_v2_T_1card
CUDA_VISIBLE_DEVICES=${cudaid1} python sa_nas_mobilenetv2.py --search_steps 1 --retain_epoch 1 --port 8881 >${log_path}/${model} 2>&1
print_info $? ${model}
# 4.2 block_sa_nas_mobilenetv2
model=block_sa_nas_v2_T_1card
CUDA_VISIBLE_DEVICES=${cudaid1} python block_sa_nas_mobilenetv2.py --search_steps 1 --port 8883 >${log_path}/${model} 2>&1
print_info $? ${model}

# 4.3 rl_nas
model=rl_nas_v2_T_1card
CUDA_VISIBLE_DEVICES=${cudaid1}  python rl_nas_mobilenetv2.py --search_steps 1 --port 8885 >${log_path}/${model} 2>&1
print_info $? ${model}
}

demo_act_det_ppyoloe(){
	cd ${slim_dir}/example/auto_compression/detection/
	wget -q https://bj.bcebos.com/v1/paddle-slim-models/detection/ppyoloe_crn_l_300e_coco.tar
	tar -xf ppyoloe_crn_l_300e_coco.tar
	wget -q https://paddle-qa.bj.bcebos.com/PaddleDetection/coco.zip
	unzip -q coco.zip

	sed -i 's/train_iter: 5000/train_iter: 30/' ./configs/ppyoloe_l_qat_dis.yaml
	sed -i 's/eval_iter: 1000/eval_iter: 10/' ./configs/ppyoloe_l_qat_dis.yaml
	sed -i 's/dataset\/coco\//coco\//g' ./configs/yolo_reader.yml

	export CUDA_VISIBLE_DEVICES=${cudaid1}
	python run.py --config_path=./configs/ppyoloe_l_qat_dis.yaml --save_dir='./act_det_demo_ppyoloe_single_card' > ${log_path}/act_det_demo_ppyoloe_single_card 2>&1
	print_info $? act_det_demo_ppyoloe_single_card

	export CUDA_VISIBLE_DEVICES=${cudaid2}
	python -m paddle.distributed.launch --log_dir=ppyoloe_log  run.py \
	--config_path=./configs/ppyoloe_l_qat_dis.yaml --save_dir='./act_det_demo_ppyoloe_multi_card' > ${log_path}/act_det_demo_ppyoloe_multi_card 2>&1
	print_info $? act_det_demo_ppyoloe_multi_card
}

demo_act_det_yolov5(){
	cd ${slim_dir}/example/auto_compression/pytorch_yolo_series
	wget -q https://bj.bcebos.com/v1/paddle-slim-models/detection/yolov5s_infer.tar
	tar -xf yolov5s_infer.tar
	wget -q https://paddle-qa.bj.bcebos.com/PaddleDetection/coco.zip
	unzip -q coco.zip

	sed -i 's/train_iter: 3000/train_iter: 30/' ./configs/yolov5s_qat_dis.yaml
	sed -i 's/eval_iter: 1000/eval_iter: 10/' ./configs/yolov5s_qat_dis.yaml
	sed -i 's/dataset\/coco\//coco\//g' ./configs/yolov5s_qat_dis.yaml

	export CUDA_VISIBLE_DEVICES=${cudaid1}
	python run.py --save_dir='./act_det_demo_yolov5s_single_card' --config_path='./configs/yolov5s_qat_dis.yaml' > ${log_path}/act_det_demo_yolov5s_single_card 2>&1
	print_info $? act_det_demo_yolov5s_single_card

	export CUDA_VISIBLE_DEVICES=${cudaid2}
	python -m paddle.distributed.launch --log_dir=yolov5s_log run.py \
          --config_path=./configs/yolov5s_qat_dis.yaml --save_dir='./act_det_demo_yolov5s_multi_card' > ${log_path}/act_det_demo_yolov5s_multi_card 2>&1
        print_info $? act_det_demo_yolov5s_multi_card
}

demo_act_det_ppyoloe_l(){
	cd ${slim_dir}/example/auto_compression/detection/
	python -m pip install paddledet

	wget -q https://bj.bcebos.com/v1/paddle-slim-models/act/ppyoloe_crn_l_300e_coco.tar
	tar -xf ppyoloe_crn_l_300e_coco.tar
	mkdir dataset && cd dataset
	wget -q https://paddle-qa.bj.bcebos.com/PaddleDetection/coco.zip
	unzip -q coco.zip
	cd ..

	sed -i 's/train_iter: 5000/train_iter: 20/' ./configs/ppyoloe_l_qat_dis.yaml
	sed -i 's/eval_iter: 1000/eval_iter: 10/' ./configs/ppyoloe_l_qat_dis.yaml

	export CUDA_VISIBLE_DEVICES=${cudaid1}
	python run.py --config_path=./configs/ppyoloe_l_qat_dis.yaml --save_dir='./act_det_demo_ppyoloe_l_single_card' > ${log_path}/act_det_demo_ppyoloe_l_single_card 2>&1
	print_info $? act_det_demo_ppyoloe_l_single_card

	export CUDA_VISIBLE_DEVICES=${cudaid2}
	python -m paddle.distributed.launch --log_dir=ppyoloe_l_log run.py \
          --config_path=./configs/ppyoloe_l_qat_dis.yaml \
          --save_dir='./act_det_demo_ppyoloe_l_mutil_card/'> ${log_path}/act_det_demo_ppyoloe_l_mutil_card 2>&1
        print_info $? act_det_demo_ppyoloe_l_mutil_card
}

demo_act_clas_MobileNetV1(){
	cd ${slim_dir}/example/auto_compression/image_classification/
	wget -q https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV1_infer.tar
	tar -xf MobileNetV1_infer.tar

	sed -i 's/eval_iter: 500/eval_iter: 50/' ./configs/MobileNetV1/qat_dis.yaml
	#sed -i 's/data_dir: .\/ILSVRC2012/data_dir: .\/data\/ILSVRC2012/' ./configs/MobileNetV1/qat_dis.yaml

	wget -q https://sys-p0.bj.bcebos.com/slim_ci/ILSVRC2012_data_demo.tar.gz --no-check-certificate
	tar xf ILSVRC2012_data_demo.tar.gz
	mv ILSVRC2012_data_demo data
	mv data/ILSVRC2012 ./

	export CUDA_VISIBLE_DEVICES=${cudaid1}
	python run.py --save_dir='./clas_demo_MobileNetV1_single_card' \
	--config_path='./configs/MobileNetV1/qat_dis.yaml' \
	> ${log_path}/act_clas_demo_MobileNetV1_single_card 2>&1
	print_info $? act_clas_demo_MobileNetV1_single_card

	export CUDA_VISIBLE_DEVICES=${cudaid2}
	python -m paddle.distributed.launch --log_dir=mobilev1_log  run.py \
		--save_dir='./clas_demo_MobileNetV1_multi_card' --config_path='./configs/MobileNetV1/qat_dis.yaml' > ${log_path}/act_clas_demo_MobileNetV1_multi_card 2>&1
	print_info $? act_clas_demo_MobileNetV1_multi_card
}

demo_act_clas_ResNet50_vd(){
	cd ${slim_dir}/example/auto_compression/image_classification/
	wget -q https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/ResNet50_vd_infer.tar
	tar -xf ResNet50_vd_infer.tar

	sed -i 's/eval_iter: 500/eval_iter: 50/' ./configs/ResNet50_vd/qat_dis.yaml
	#sed -i 's/data_dir: .\/ILSVRC2012/data_dir: .\/data\/ILSVRC2012/' ./configs/ResNet50_vd/qat_dis.yaml

	export CUDA_VISIBLE_DEVICES=${cudaid1}
	python run.py --save_dir='./clas_demo_ResNet50_vd_single_card' --config_path='./configs/ResNet50_vd/qat_dis.yaml' > ${log_path}/act_clas_demo_ResNet50_vd_single_card 2>&1
	print_info $? act_clas_demo_ResNet50_vd_single_card
	export CUDA_VISIBLE_DEVICES=${cudaid2}
	python -m paddle.distributed.launch --log_dir=mobilev1_log  run.py \
		--save_dir='./clas_demo_ResNet50_vd_multi_card' --config_path='./configs/ResNet50_vd/qat_dis.yaml' > ${log_path}/act_clas_demo_ResNet50_vd_multi_card 2>&1
	print_info $? act_clas_demo_ResNet50_vd_multi_card
}

demo_act_clas_MobileNetV3(){
	cd ${slim_dir}/example/auto_compression/image_classification/
	wget -q https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV3_large_x1_0_infer.tar
  tar -xf MobileNetV3_large_x1_0_infer.tar

	sed -i 's/eval_iter: 5000/eval_iter: 10/' ./configs/MobileNetV3_large_x1_0/qat_dis.yaml
	sed -i 's/epochs: 2/epochs: 1/' ./configs/MobileNetV3_large_x1_0/qat_dis.yaml

	export CUDA_VISIBLE_DEVICES=${cudaid1}
	python run.py --save_dir='./save_quant_mobilev3_single_card/' \
	--config_path='./configs/MobileNetV3_large_x1_0/qat_dis.yaml' \
	> ${log_path}/act_clas_demo_mobilev3_single_card 2>&1
	print_info $? act_clas_demo_mobilev3_single_card

	export CUDA_VISIBLE_DEVICES=${cudaid2}
	python -m paddle.distributed.launch run.py \
	--save_dir='./save_quant_mobilev3_mutli_card/' \
	--config_path='./configs/MobileNetV3_large_x1_0/qat_dis.yaml' \
	> ${log_path}/act_clas_demo_mobilev3_multi_card 2>&1
  print_info $? act_clas_demo_mobilev3_multi_card
}

demo_act_nlp_pp_minilm(){
	cd ${slim_dir}/example/auto_compression/nlp/
	wget -q https://bj.bcebos.com/v1/paddle-slim-models/act/afqmc.tar
	tar -xf afqmc.tar

	sed -i 's/epochs: 6/train_iter: 100/' ./configs/pp-minilm/auto/afqmc.yaml
	sed -i 's/eval_iter: 1070/eval_iter: 100/' ./configs/pp-minilm/auto/afqmc.yaml
	sed -i 's/HyperParameterOptimization:/#HyperParameterOptimization:/' ./configs/pp-minilm/auto/afqmc.yaml
	sed -i 's/QuantPost:/#QuantPost:/' ./configs/pp-minilm/auto/afqmc.yaml

	export CUDA_VISIBLE_DEVICES=${cudaid1}
	python run.py --config_path='./configs/pp-minilm/auto/afqmc.yaml' --save_dir='./save_afqmc_pp_minilm_pruned' > ${log_path}/act_nlp_demo_pp_minilm_single_card 2>&1
	print_info $? act_nlp_demo_pp_minilm_single_card
	sed -i 's/.\/afqmc/.\/save_afqmc_pp_minilm_pruned/' ./configs/pp-minilm/auto/afqmc.yaml
	python run.py --config_path='./configs/pp-minilm/auto/afqmc.yaml'  --eval True > ${log_path}/act_nlp_demo_pp_minilm_single_card_eval 2>&1
	print_info $? act_nlp_demo_pp_minilm_single_card_eval
}

demo_act_nlp_ERNIE_3(){
	cd ${slim_dir}/example/auto_compression/nlp/
	wget -q https://bj.bcebos.com/v1/paddle-slim-models/act/NLP/ernie3.0-medium/fp32_models/AFQMC.tar
	tar -xf AFQMC.tar

	sed -i 's/epochs: 6/train_iter: 100/' ./configs/ernie3.0/afqmc.yaml
	sed -i 's/eval_iter: 1070/eval_iter: 100/' ./configs/ernie3.0/afqmc.yaml

	export CUDA_VISIBLE_DEVICES=${cudaid1}
	python run.py --config_path='./configs/ernie3.0/afqmc.yaml' --save_dir='./save_afqmc_ERNIE_pruned' > ${log_path}/act_nlp_demo_ernie_3_single_card 2>&1
	print_info $? act_nlp_demo_ernie_3_single_card
	sed -i 's/.\/afqmc/.\/save_afqmc_ERNIE_pruned/' ./configs/pp-minilm/auto/afqmc.yaml
	python run.py --config_path='./configs/ernie3.0/afqmc.yaml'  --eval True > ${log_path}/act_nlp_demo_ernie3_single_card_eval 2>&1
	print_info $? act_nlp_demo_ernie3_single_card_eval
}

demo_act_seg_pp_Liteseg_qat(){
	cd ${slim_dir}/example/auto_compression/semantic_segmentation/
	cd data
	wget -q https://bj.bcebos.com/v1/paddle-slim-models/data/mini_cityscapes/mini_cityscapes.tar
	tar -xf mini_cityscapes.tar
	mv mini_cityscapes cityscapes
	cd ..

	wget -q https://paddleseg.bj.bcebos.com/tipc/easyedge/RES-paddle2-PPLIteSegSTDC1.zip
	unzip -q RES-paddle2-PPLIteSegSTDC1.zip
	ls

	sed -i 's/epochs: 20/epochs: 1/' ./configs/pp_liteseg/pp_liteseg_qat.yaml
	sed -i '/epochs: 1/a\  train_iter: 100'  ./configs/pp_liteseg/pp_liteseg_qat.yaml
	sed -i 's/eval_iter: 180/eval_iter: 100/' ./configs/pp_liteseg/pp_liteseg_qat.yaml

	export CUDA_VISIBLE_DEVICES=${cudaid1}
	python run.py \
    --config_path='configs/pp_liteseg/pp_liteseg_qat.yaml' \
    --save_dir='./save_pp_lite_seg_model_qat_single_card'  > ${log_path}/act_seg_demo_pp_Liteseg_qat_single_card 2>&1
	print_info $? act_seg_demo_pp_Liteseg_qat_single_card

	export CUDA_VISIBLE_DEVICES=${cudaid2}
	python -m paddle.distributed.launch --log_dir=pp_Liteseg_qat_log run.py \
	--config_path='configs/pp_liteseg/pp_liteseg_qat.yaml' \
    --save_dir='./save_pp_lite_seg_model_qat_multi_card'  > ${log_path}/act_seg_demo_pp_Liteseg_qat_multi_card 2>&1
	print_info $? act_seg_demo_pp_Liteseg_qat_multi_card
}

demo_act_seg_pp_Liteseg_auto(){
	cd ${slim_dir}/example/auto_compression/semantic_segmentation/

	sed -i 's/epochs: 14/epochs: 1/' ./configs/pp_liteseg/pp_liteseg_auto.yaml
	sed -i '/epochs: 1/a\ train_iter: 100'  ./configs/pp_liteseg/pp_liteseg_auto.yaml

	export CUDA_VISIBLE_DEVICES=${cudaid1}
	python run.py \
    --config_path='configs/pp_liteseg/pp_liteseg_qat.yaml' \
    --save_dir='./save_pp_lite_seg_model_auto_single_card'  > ${log_path}/act_seg_demo_pp_Liteseg_auto_single_card 2>&1
	print_info $? act_seg_demo_pp_Liteseg_auto_single_card

	export CUDA_VISIBLE_DEVICES=${cudaid2}
	python -m paddle.distributed.launch --log_dir=pp_Liteseg_auto_log run.py \
	--config_path='configs/pp_liteseg/pp_liteseg_qat.yaml' \
    --save_dir='./save_pp_lite_seg_model_auto_multi_card'  > ${log_path}/act_seg_demo_pp_Liteseg_auto_multi_card 2>&1
	print_info $? act_seg_demo_pp_Liteseg_auto_multi_card
}


demo_act_seg_pp_Liteseg_sparse(){
	cd ${slim_dir}/example/auto_compression/semantic_segmentation/

	sed -i 's/epochs: 50/epochs: 1/' ./configs/pp_liteseg/pp_liteseg_sparse.yaml
	sed -i '/epochs: 1/a\  train_iter: 100'  ./configs/pp_liteseg/pp_liteseg_sparse.yaml
	sed -i '/eval_iter: 180/eval_iter: 100/'  ./configs/pp_liteseg/pp_liteseg_sparse.yaml

	export CUDA_VISIBLE_DEVICES=${cudaid1}
	python run.py \
    --save_dir='./save_pp_lite_seg_model_sparse_single_card' \
    --config_path='configs/pp_liteseg/pp_liteseg_qat.yaml'  > ${log_path}/act_seg_demo_pp_Liteseg_sparse_single_card 2>&1
	print_info $? act_seg_demo_pp_Liteseg_sparse_single_card

	export CUDA_VISIBLE_DEVICES=${cudaid2}
	python -m paddle.distributed.launch --log_dir=pp_Liteseg_sparse_log run.py \
	--config_path='configs/pp_liteseg/pp_liteseg_qat.yaml' \
    --save_dir='./save_pp_lite_seg_model_sparse_multi_card'  > ${log_path}/act_seg_demo_pp_Liteseg_sparse_multi_card 2>&1
	print_info $? act_seg_demo_pp_Liteseg_sparse_multi_card
}

all_act_CI(){
  demo_act_clas_MobileNetV1
  demo_act_nlp_pp_minilm
  demo_act_seg_pp_Liteseg_qat
}

all_act_CE(){
  demo_act_det_ppyoloe_l
  demo_act_det_ppyoloe
  demo_act_det_yolov5
  demo_act_clas_MobileNetV1
  demo_act_clas_MobileNetV3
  demo_act_clas_ResNet50_vd
  demo_act_nlp_pp_minilm
  demo_act_nlp_ERNIE_3
  demo_act_seg_pp_Liteseg_qat
  demo_act_seg_pp_Liteseg_auto
  demo_act_seg_pp_Liteseg_sparse

}

all_act_ALL(){
  demo_act_det_ppyoloe_l
  demo_act_det_ppyoloe
  demo_act_det_yolov5
  demo_act_clas_MobileNetV1
  demo_act_clas_ResNet50_vd
  demo_act_clas_MobileNetV3
  demo_act_nlp_pp_minilm
  demo_act_nlp_ERNIE_3
  demo_act_seg_pp_Liteseg_qat
  demo_act_seg_pp_Liteseg_auto
  demo_act_seg_pp_Liteseg_sparse

}

demo_full_quant_clas_MobileNetV3(){
	cd ${slim_dir}/example/full_quantization/image_classification/
  wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV3_large_x1_0_infer.tar
  tar -xf MobileNetV3_large_x1_0_infer.tar

  wget -q https://sys-p0.bj.bcebos.com/slim_ci/ILSVRC2012_data_demo.tar.gz --no-check-certificate
	tar xf ILSVRC2012_data_demo.tar.gz
	mv ILSVRC2012_data_demo data
	mv data/ILSVRC2012 ./

	sed -i 's/epochs: 2/epochs: 1/' ./configs/mobilenetv3_large_qat_dis.yaml
	sed -i 's/eval_iter: 5000/eval_iter: 10/' ./configs/mobilenetv3_large_qat_dis.yaml

	export CUDA_VISIBLE_DEVICES=${cudaid1}
	python run.py --save_dir='./full_quant_save_quant_mobilev3_single_card/' \
	--config_path='./configs/mobilenetv3_large_qat_dis.yaml' \
	> ${log_path}/full_quant_clas_demo_mobilev3_single_card 2>&1
	print_info $? full_quant_clas_demo_mobilev3_single_card

	export CUDA_VISIBLE_DEVICES=${cudaid2}
	python -m paddle.distributed.launch run.py \
	--save_dir='./full_quant_save_quant_mobilev3_multi_card/' \
	--config_path='./configs/mobilenetv3_large_qat_dis.yaml' \
	> ${log_path}/full_quant_clas_demo_mobilev3_multi_card 2>&1
	print_info $? full_quant_clas_demo_mobilev3_multi_card
}

demo_full_quant_det_picodet(){
	cd ${slim_dir}/example/full_quantization/picodet/
	wget -q https://bj.bcebos.com/v1/paddle-slim-models/act/picodet_s_416_coco_npu.tar
  tar -xf picodet_s_416_coco_npu.tar

  mkdir dataset
  cd dataset
  wget -q https://paddle-qa.bj.bcebos.com/PaddleDetection/coco.zip
	unzip -q coco.zip
	cd ..

	sed -i 's/train_iter: 8000/train_iter: 20/' ./configs/picodet_npu_with_postprocess.yaml
	sed -i 's/eval_iter: 1000/eval_iter: 10/' ./configs/picodet_npu_with_postprocess.yaml

	export CUDA_VISIBLE_DEVICES=${cudaid1}
	python run.py --config_path=./configs/picodet_npu_with_postprocess.yaml \
	--save_dir='./full_quant_det_picodet_single_card/' \
	> ${log_path}/full_quant_det_picodet_single_card 2>&1
	print_info $? full_quant_det_picodet_single_card

	export CUDA_VISIBLE_DEVICES=${cudaid2}
	python -m paddle.distributed.launch --log_dir=log run.py \
	--save_dir='./full_quant_det_picodet_multi_card/' \
	--config_path=./configs/picodet_npu_with_postprocess.yaml \
	> ${log_path}/full_quant_det_picodet_multi_card 2>&1
	print_info $? full_quant_det_picodet_multi_card
}

all_full_quant_CE(){
  demo_full_quant_clas_MobileNetV3
  demo_full_quant_det_picodet
}

run_case_func(){
    echo --- start run case ---
    case_num=1
    for model in ${all_case_list[*]};do
    	{
        echo ---$case_num/${#all_case_list[*]}: ${model}---
        ${model}
	}
        let case_num++
    done
    echo --- end run case---
}


print_logs_func(){
    cd ${log_path}
    FF=`ls *FAIL*|wc -l`
    if [ "${FF}" -gt "0" ];then
        echo ---fail case: ${FF}
        ls *FAIL*
        exit ${FF}
    else
        echo ---all case pass---
        exit 0
    fi
}

python -m pip install paddleclas
python -m pip install paddlenlp
python -m pip install paddleseg==2.5.0
python -m pip install paddledet
python -m pip list | grep paddle

if [ "$1" = "run_CI" ];then
	# CI任务的case
    export all_case_list=(all_act_CI all_st_quant_CI all_distill_CI  all_dy_quant_CI all_st_prune_CI all_dy_prune_CI all_st_unstr_prune_CI all_dy_unstr_prune_CI )
    run_case_func
    print_logs_func

elif [ "$1" = "run_CE" ];then
	# CE任务的case
    export all_case_list=(all_full_quant_CE all_act_CE all_st_quant_CE all_dy_quant_CE all_st_prune_CE all_dy_prune_CE all_st_unstr_prune_CE all_dy_unstr_prune_CE demo_sa_nas )
    run_case_func
     wait
    # print_logs_func
elif [ "$1" = "run_ALL" ];then
	# 全量case
    export all_case_list=(all_full_quant_CE all_distill_ALL all_st_quant_ALL all_dy_quant_ALL all_st_prune_ALL all_dy_prune_ALL all_st_unstr_prune_ALL all_dy_unstr_prune_ALL demo_sa_nas all_act_ALL)
    run_case_func
    wait
    run_case_without_multiprocess
else
    echo "slim's $1 case is running"
    $1
fi
