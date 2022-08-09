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

python distill.py --num_epochs 1 --save_inference True > ${log_path}/st_distill_ResNet50_vd_MobileNet 2>&1
print_info $? st_distill_ResNet50_vd_MobileNet
}

demo_distill_02(){
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
--checkpoint_dir ./output/mobilenetv1 --num_epochs 1 --batch_size 32 >${log_path}/st_quant_aware_v1 2>&1
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
--last_index 0 --emb_quant True >${log_path}/st_quant_em_afteri_nfer 2>&1
print_info $? st_quant_em_afteri_nfer
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
python ../quant_post/eval.py \
--model_path=./inference_model/MobileNetV1_quant/ > ${log_path}/st_quant_post_hpo_eval 2>&1
print_info $? st_quant_post_hpo_eval
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
    demo_st_quant_aware_v1
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

    echo "---eval qat int8_infer : ${model}---"
    python ./src/test.py \
   	--model_path=int8_qat_models/${model}  \
   	--data_dir=${data_path} \
   	--test_samples=-1 \
   	--batch_size=32 \
   	--use_gpu=False \
   	--ir_optim=False  > ${log_path}/dy_qat_eval_int8_${model} 2>&1
    print_info $? dy_qat_eval_int8_${model}
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

    echo "-------- eval int8_infer model -------------", ${model}
    python ./src/test.py \
        --model_path=${output_dir}/${model}/int8_infer \
        --data_dir=${data_path} \
        --batch_size=${batch_size} \
        --use_gpu=False \
        --test_samples=${test_samples} \
        --ir_optim=False > ${log_path}/dy_ptq_eval_int8_${model} 2>&1
        print_info $? dy_ptq_eval_int8_${model}
done
}


all_dy_quant_CI(){
    demo_dy_quant_v1
}

all_dy_quant_CE(){
    demo_dy_quant_v3
    ce_tests_dy_qat4
    ce_tests_dy_ptq4
}

all_dy_quant_ALL(){
    demo_dy_quant_v1
    demo_dy_quant_v3
    ce_tests_dy_qat4
    ce_tests_dy_ptq4
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
          --batch_size 64 \
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
    demo_dy_unstructured_pruning_ratio_gmp
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

demo_auto_mbv2_qat_dis(){
cd ${slim_dir}/demo/auto_compression/
wget -q https://paddle-qa.bj.bcebos.com/PaddleSlim_datasets/infermodel_mobilenetv2.tar
tar -xf infermodel_mobilenetv2.tar

python demo_imagenet.py \
    --model_dir='infermodel_mobilenetv2' \
    --model_filename='inference.pdmodel' \
    --params_filename='./inference.pdiparams' \
    --save_dir='./save_qat_mbv2/' \
    --devices='gpu' \
    --batch_size=128 \
    --data_dir='../data/ILSVRC2012/' \
    --config_path='./configs/CV/mbv2_qat_dis.yaml' > ${log_path}/auto_mbv2_qat_dis 2>&1
print_info $? auto_mbv2_qat_dis
}

demo_auto_mbv2_ptq_hpo(){
cd ${slim_dir}/demo/auto_compression/
sed -i 's/max_quant_count: 20/max_quant_count: 1/' ./configs/CV/mbv2_ptq_hpo.yaml

python demo_imagenet.py \
    --model_dir='infermodel_mobilenetv2' \
    --model_filename='inference.pdmodel' \
    --params_filename='./inference.pdiparams' \
    --save_dir='./save_ptq_mbv2/' \
    --devices='gpu' \
    --batch_size=128 \
    --data_dir='../data/ILSVRC2012/' \
    --config_path='./configs/CV/mbv2_ptq_hpo.yaml' > ${log_path}/auto_mbv2_ptq_hpo 2>&1
print_info $? auto_mbv2_ptq_hpo
}

demo_auto_bert_asp_dis(){
CUDA_VISIBLE_DEVICES=${cudaid1}
cd ${slim_dir}/demo/auto_compression/
sed -i 's/epochs: 3/epochs: 1/' ./configs/NLP/bert_asp_dis.yaml
wget -q https://paddle-qa.bj.bcebos.com/PaddleSlim_datasets/static_bert_models.tar.gz
tar -xf static_bert_models.tar.gz
python -m pip install paddlenlp

python demo_glue.py \
    --model_dir='./static_bert_models/' \
    --model_filename='bert.pdmodel' \
    --params_filename='bert.pdiparams' \
    --save_dir='./save_asp_bert/' \
    --devices='gpu' \
    --batch_size=64 \
    --task='sst-2' \
    --config_path='./configs/NLP/bert_asp_dis.yaml' > ${log_path}/auto_bert_asp_dis 2>&1
print_info $? auto_bert_asp_dis
}

all_auto_CE(){
    demo_auto_mbv2_qat_dis
    demo_auto_mbv2_ptq_hpo
    demo_auto_bert_asp_dis
}

all_auto_ALL(){
    demo_auto_mbv2_qat_dis
    demo_auto_mbv2_ptq_hpo
    demo_auto_bert_asp_dis
}

if [ "$1" = "run_CI" ];then
	# CI任务的case
    export all_case_list=(all_st_quant_CI all_distill_CI  all_dy_quant_CI all_st_prune_CI all_dy_prune_CI all_st_unstr_prune_CI all_dy_unstr_prune_CI )
elif [ "$1" = "run_CE" ];then
	# CE任务的case、暂时去掉all_auto_CE
    export all_case_list=(all_st_quant_CE all_dy_quant_CE all_st_prune_CE all_dy_prune_CE all_st_unstr_prune_CE all_dy_unstr_prune_CE demo_sa_nas)
elif [ "$1" = "ALL" ];then
	# 全量case、暂时去掉all_auto_ALL
    export all_case_list=(all_distill_ALL all_st_quant_ALL all_dy_quant_ALL all_st_prune_ALL all_dy_prune_ALL all_st_unstr_prune_ALL all_dy_unstr_prune_ALL demo_sa_nas)
fi

echo --- start run case ---
case_num=1
for model in ${all_case_list[*]};do
    echo ---$case_num/${#all_case_list[*]}: ${model}---
    ${model}
    let case_num++
done
echo --- end run case---
