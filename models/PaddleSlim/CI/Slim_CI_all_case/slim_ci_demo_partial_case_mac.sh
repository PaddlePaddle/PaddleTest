#!/usr/bin/env bash
##################
echo "run slim demo task of mac with paramters" $1, $2
export slim_dir=$PWD/PaddleSlim;
# for logs env
if [ -d "logs" ];then
    rm -rf logs;
fi
mkdir logs
export log_path=$PWD/logs;
cd ${slim_dir}
# for paddleslim env
slim1_install (){
    echo -e "\033[35m ---- only install slim \033[0m"
    python -m pip install -U paddleslim
}
slim2_build (){
    echo -e "\033[35m ---- build and install slim  \033[0m"
    python -m pip install matplotlib
    python -m pip install -r requirements.txt
    python setup.py install
}
slim3_build_whl (){
    echo -e "\033[35m ---- build and install slim  \033[0m"
    python -m pip install matplotlib
    python -m pip install -r requirements.txt
    python setup.py bdist_wheel
    python -m pip install dist/*.whl;
}
$1
python -m pip list;
####################################
print_info(){
if [ $1 -ne 0 ];then
    mv ${log_path}/$2 ${log_path}/F_$2.log
    echo -e "\033[31m ${log_path}/F_$2 \033[0m"
else
    mv ${log_path}/$2 ${log_path}/S_$2.log
    echo -e "\033[32m ${log_path}/S_$2 \033[0m"
fi
}

# UT
UT_test(){
if [ -d "ut_logs" ];then
    rm -rf ut_logs;
fi
mkdir ut_logs
test_num=1
for line in `ls test_*.py`
do
    name=`echo ${line} | cut -d \. -f 1`
    python  ${line} > ut_logs/${name}.log 2>&1
    if [ $? -ne 0 ];then
      mv ut_logs/${name}.log  ut_logs/F_${test_num}_${name}.log
      echo -e "\033[31m ut_logs/F_${test_num}_${name} \033[0m"
    else
      mv ut_logs/${name}.log ut_logs/S_${test_num}_${name}.log
      echo -e "\033[32m ut_logs/S_${test_num}_${name} \033[0m"
    fi
    let test_num++
done
}

run_UT (){
echo --------- run UT -----------
cd ${slim_dir}/tests
UT_test
mv ut_logs ${log_path}/st_ut_logs

cd ${slim_dir}/tests/dygraph
UT_test
mv ut_logs ${log_path}/dy_ut_logs
}

if [ $2 == True ];then
  run_UT
else
  echo ------- skip run UT -------
fi

catchException() {
  echo $1 failed due to exception >> FAIL_Exception.log
}

# download dataset: PaddleSlim/demo/data/ILSVRC2012
cd ${slim_dir}/demo
if [ -d "data" ];then
    rm -rf data
fi
wget -q https://sys-p0.bj.bcebos.com/slim_ci/ILSVRC2012_data_demo.tar.gz --no-check-certificate
tar xf ILSVRC2012_data_demo.tar.gz
mv ILSVRC2012_data_demo data
# download pretrain model:PaddleSlim/demo/pretrain
root_url="http://paddle-imagenet-models-name.bj.bcebos.com"
pre_models="MobileNetV1 MobileNetV2 MobileNetV3_large_x1_0_ssld ResNet101_vd MobileNetV2 ResNet34 ResNet50 ResNet50_vd"
if [ -d "pretrain" ];then
    rm -rf pretrain
fi
mkdir pretrain && cd pretrain
for model in ${pre_models}
do
    if [ ! -f ${model} ]; then
        wget -q ${root_url}/${model}_pretrained.tar
        tar xf ${model}_pretrained.tar
    fi
done

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
demo_deep_mutual_learning(){
cd ${slim_dir}/demo/deep_mutual_learning || catchException demo_deep_mutual_learning
python dml_train.py --epochs 1 --use_gpu False > ${log_path}/dml_mv1_mv1_gpu1 2>&1
print_info $? dml_mv1_mv1_gpu1
}

all_distillation(){
demo_distillation
#demo_deep_mutual_learning
}

#2.quant
# 2.1 quant/quant_aware 使用小数据集即可
demo_quant_aware(){
cd ${slim_dir}/demo/quant/quant_aware || catchException demo_quant_aware
if [ -d "output" ];then
    rm -rf output
fi
# 2.1版本时默认BS=256会报显存不足，故暂时修改成128
python train.py --model MobileNet \
--pretrained_model ../../pretrain/MobileNetV1_pretrained \
--checkpoint_dir ./output/mobilenetv1 --num_epochs 1 --batch_size 128 --use_gpu False >${log_path}/quant_aware_v1_T 2>&1
print_info $? quant_aware_v1_T
}

demo_quant_embedding(){
cd ${slim_dir}/demo/quant/quant_embedding || catchException demo_quant_embedding
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
 --base_lr 1.0 --print_batch 1000 --with_speed --is_sparse >${log_path}/quant_em_word2vec_T 2>&1
print_info $? quant_em_word2vec_T
}

# 2.3 quan_post # 小数据集
demo_st_quant_post(){
# 20210425 新增4种离线量化方法
cd ${slim_dir}/demo/quant/quant_post
# 1 导出模型
python export_model.py --model "MobileNet" --use_gpu False --pretrained_model ../../pretrain/MobileNetV1_pretrained \
--data imagenet >${log_path}/st_quant_post_v1_export 2>&1
print_info $? st_quant_post_v1_export

# 3 离线量化
# 4 量化后eval
for algo in hist
do
## 不带bc 离线量化
echo "quant_post train no bc " ${algo}
python quant_post.py --model_path ./inference_model/MobileNet \
--save_path ./quant_model/${algo}/MobileNet \
--model_filename model --params_filename weights --algo ${algo} --use_gpu False >${log_path}/st_quant_post_v1_T_${algo} 2>&1
print_info $? st_quant_post_v1_T_${algo}
# 量化后eval
echo "quant_post eval no bc " ${algo}
python eval.py --model_path ./quant_model/${algo}/MobileNet --model_name __model__ \
--params_name __params__ --use_gpu False > ${log_path}/st_quant_post_${algo}_eval2 2>&1
print_info $? st_quant_post_${algo}_eval2

# 带bc参数的 离线量化
echo "quant_post train bc " ${algo}
python quant_post.py --model_path ./inference_model/MobileNet \
--save_path ./quant_model/${algo}_bc/MobileNet \
--model_filename model --params_filename weights \
--algo ${algo} --bias_correction True --use_gpu False >${log_path}/st_quant_post_T_${algo}_bc 2>&1
print_info $? st_quant_post_T_${algo}_bc

# 量化后eval
echo "quant_post eval bc " ${algo}
python eval.py --model_path ./quant_model/${algo}_bc/MobileNet --model_name __model__ \
--params_name __params__ --use_gpu False > ${log_path}/st_quant_post_${algo}_bc_eval2 2>&1
print_info $? st_quant_post_${algo}_bc_eval2

done
}

#2.4
demo_quant_pact_quant_aware(){
cd ${slim_dir}/demo/quant/pact_quant_aware
python train.py --model MobileNetV3_large_x1_0 \
--pretrained_model ../../pretrain/MobileNetV3_large_x1_0_ssld_pretrained \
--num_epochs 1 --lr 0.0001 --use_pact True --batch_size 128 --lr_strategy=piecewise_decay \
--step_epochs 2 --l2_decay 1e-5 --use_gpu False >${log_path}/demo_quant_pact_quant_aware_v3 2>&1
print_info $? demo_quant_pact_quant_aware_v3
# load
python train.py --model MobileNetV3_large_x1_0 \
--pretrained_model ../../pretrain/MobileNetV3_large_x1_0_ssld_pretrained \
--num_epochs 2 --lr 0.0001 --use_pact True --batch_size 128 --lr_strategy=piecewise_decay \
--step_epochs 20 --l2_decay 1e-5 \
--checkpoint_dir ./output/MobileNetV3_large_x1_0/0 \
--checkpoint_epoch 0 --use_gpu False >${log_path}/demo_quant_pact_quant_aware_v3_load 2>&1
print_info $? demo_quant_pact_quant_aware_v3_load
}

# 2.5
demo_dygraph_quant(){
cd ${slim_dir}/demo/dygraph/quant
python train.py --model='mobilenet_v1' \
--pretrained_model '../../pretrain/MobileNetV1_pretrained' \
--num_epochs 1 \
--batch_size 128 --use_gpu False > ${log_path}/dy_quant_v1 2>&1
print_info $? dy_quant_v1
}
# 2.6

demo_dy_qat1(){
cd ${slim_dir}/ce_tests/dygraph/quant
ln -s ${slim_dir}/demo/data/ILSVRC2012
test_samples=1000  # if set as -1, use all test samples
data_path='./ILSVRC2012/'
batch_size=16
epoch=1
lr=0.0001
num_workers=1
output_dir=$PWD/output_models
for model in mobilenet_v1
do
#        echo "------1 nopact train--------", ${model}
#        python ./src/qat.py \
#        --arch=${model} \
#        --data=${data_path} \
#        --epoch=${epoch} \
#        --batch_size=32 \
#        --num_workers=${num_workers} \
#        --lr=${lr} \
#        --output_dir=${output_dir} \
#        --enable_quant \
#        --use_gpu False > qat_${model}_gpu1_nw1 2>&1
#        # 2 eval before save quant
#        echo "--------2 eval before save quant -------------", ${model}
#        python ./src/eval.py \
#        --model_path=./output_models/quant_dygraph/${model} \
#        --data_dir=${data_path} \
#        --test_samples=${test_samples} \
#        --batch_size=${batch_size} --use_gpu False > eval_before_save_${model} 2>&1
#        # 3 CPU上部署量化模型,需要使用`test/save_quant_model.py`脚本进行模型转换。
#        echo "--------3 save_nopact_quant_model-------------", ${model}
#        python src/save_quant_model.py \
#          --load_model_path output_models/quant_dygraph/${model} \
#          --save_model_path int8_models/${model} > save_quant_${model} 2>&1
#        # 4
#        echo "--------4 CPU eval after save nopact quant -------------", ${model}
#        python ./src/eval.py \
#        --model_path=./int8_models/${model} \
#        --data_dir=${data_path} \
#        --test_samples=${test_samples} \
#        --batch_size=${batch_size} > cpu_eval_after_save_${model} 2>&1
#    elif [ $1 == pact ];then
    # 1 pact quant train
        echo "------use pact train--------", ${model}
        python ./src/qat.py \
        --arch=${model} \
        --data=${data_path} \
        --epoch=${epoch} \
        --batch_size=32 \
        --num_workers=${num_workers} \
        --lr=${lr} \
        --output_dir=$PWD/output_models_pact/ \
        --enable_quant \
        --use_pact  > pact_qat_${model}_gpu1_nw1 2>&1
        # 2 eval before save quant
        echo "--------2 eval before save pact quant -------------", ${model}
        python ./src/eval.py \
        --model_path=./output_models_pact/quant_dygraph/${model} \
        --data_dir=${data_path} \
        --test_samples=${test_samples} \
        --batch_size=${batch_size}  > eval_before_pact_save_${model} 2>&1
        echo "--------3  save pact quant -------------", ${model}
        python src/save_quant_model.py \
          --load_model_path output_models_pact/quant_dygraph/${model} \
          --save_model_path int8_models_pact/${model} > save_pact_quant_${model} 2>&1
        echo "--------4 CPU eval after save pact quant -------------", ${model}
        python ./src/eval.py \
        --model_path=./int8_models_pact/${model} \
        --data_dir=${data_path} \
        --test_samples=${test_samples} \
        --batch_size=${batch_size} > cpu_eval_after_pact_save_${model} 2>&1
#    fi

done
}

ce_tests_dygraph_ptq4(){
cd ${slim_dir}/ce_tests/dygraph/quant || catchException ce_tests_dygraph_ptq4
ln -s ${slim_dir}/demo/data/ILSVRC2012
test_samples=1000  # if set as -1, use all test samples
data_path='./ILSVRC2012/'
batch_size=32
epoch=1
output_dir="./output_ptq"
quant_batch_num=10
quant_batch_size=10
#for model in mobilenet_v1 mobilenet_v2 resnet50 vgg16
for model in mobilenet_v1
do
    echo "--------quantize model: ${model}-------------"
    # save ptq quant model
    python ./src/ptq.py \
        --data=${data_path} \
        --arch=${model} \
        --quant_batch_num=${quant_batch_num} \
        --quant_batch_size=${quant_batch_size} \
        --output_dir=${output_dir} > ${log_path}/ptq_${model} 2>&1
        print_info $? ptq_${model}

    echo "-------- eval fp32_infer model -------------", ${model}
    python ./src/test.py \
        --model_path=${output_dir}/${model}/fp32_infer \
        --data_dir=${data_path} \
        --batch_size=${batch_size} \
        --use_gpu=False \
        --test_samples=${test_samples} \
        --ir_optim=False > ${log_path}/ptq_eval_fp32_${model} 2>&1
        print_info $? ptq_eval_fp32_${model}

    echo "-------- eval int8_infer model -------------", ${model}
    python ./src/test.py \
        --model_path=${output_dir}/${model}/int8_infer \
        --data_dir=${data_path} \
        --batch_size=${batch_size} \
        --use_gpu=False \
        --test_samples=${test_samples} \
        --ir_optim=False > ${log_path}/ptq_eval_int8_${model} 2>&1
        print_info $? ptq_eval_int8_${model}

done
}

#用于更新release分支下无ce_tests_dygraph_ptq case；release分支设置is_develop="False"
is_develop="True"

all_quant(){
  if [ "${is_develop}" == "True" ];then
      ce_tests_dygraph_ptq4
  fi
  demo_quant_aware
  demo_quant_embedding
  demo_st_quant_post
  demo_quant_pact_quant_aware
  demo_dygraph_quant
  demo_dy_qat1

}

# 3 prune
# 3.1 P0 prune
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

# 3.4 dygraph_prune
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

#2.4. 评估  通过调用eval.py脚本，对剪裁和重训练后的模型在测试数据上进行精度：
python eval.py \
--checkpoint=./fpgm_resnet34_025_120_models/1 \
--model="resnet34" \
--pruned_ratio=0.25 \
--batch_size=128 --use_gpu False >${log_path}/dy_prune_ResNet50_f42_eval 2>&1
print_info $? dy_prune_ResNet50_f42_eval

#2.5. 导出模型   执行以下命令导出用于预测的模型：
python export_model.py \
--checkpoint=./fpgm_resnet34_025_120_models/final \
--model="resnet34" \
--pruned_ratio=0.25 \
--output_path=./infer_final/resnet  > ${log_path}/dy_prune_ResNet50_f42_export 2>&1
print_info $? dy_prune_ResNet50_f42_export
}

# 3.5 unstructured_prune
demo_st_unstructured_prune(){
cd ${slim_dir}/demo/unstructured_prune || catchException demo_st_unstructured_prune

# MNIST数据集
python train.py \
--batch_size 128 \
--pretrained_model ../pretrain/MobileNetV1_pretrained \
--lr 0.05 \
--pruning_mode threshold \
--threshold 0.01 \
--data mnist \
--lr_strategy piecewise_decay \
--step_epochs 1 2 3 \
--num_epochs 1 \
--test_period 1 \
--model_path st_unstructured_models_mnist --use_gpu False >${log_path}/st_unstructured_prune_threshold_mnist_T 2>&1
print_info $? st_unstructured_prune_threshold_mnist_T
# eval
python evaluate.py \
       --pruned_model=st_unstructured_models_mnist \
       --data="mnist"  --use_gpu False >${log_path}/st_unstructured_prune_threshold_mnist_eval &
print_info $? st_unstructured_prune_threshold_mnist_eval
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
# eval
python evaluate.py --pruned_model dy_threshold_models/model-pruned.pdparams \
--data imagenet --use_gpu False >${log_path}/dy_threshold_prune_eval 2>&1
print_info $? dy_threshold_prune_eval
# cifar10
#python train.py --data cifar10 --lr 0.05 \
#--pruning_mode threshold \
#--threshold 0.01 --use_gpu False >${log_path}/dy_unstructured_prune_threshold_prune_cifar10_T 2>&1
#print_info $? dy_unstructured_prune_threshold_prune_cifar10_T

}

all_prune(){
  demo_prune_v1
  demo_prune_fpgm_v2_T
  demo_dy_prune_ResNet34_f42
  demo_st_unstructured_prune
  demo_dy_unstructured_prune
}

#4 nas
# 4.1 sa_nas_mobilenetv2
demo_nas(){
cd ${slim_dir}/demo/nas || catchException demo_nas
python sa_nas_mobilenetv2.py --search_steps 1 --port 8881 --use_gpu False >${log_path}/sa_nas_v2_T_1card 2>&1
print_info $? sa_nas_v2_T_1card

# 4.4 parl_nas
#model=parl_nas_v2_T_1card
#CUDA_VISIBLE_DEVICES=${cudaid1} python parl_nas_mobilenetv2.py \
#--search_steps 1 --port 8887 >${log_path}/${model} 2>&1
#print_info $? ${model}
}
all_nas(){ # 3 个模型
    demo_nas
}

# 5 darts
# search 1card # DARTS一阶近似搜索方法
darts_1(){
cd ${slim_dir}/demo/darts || catchException darts_1
model=darts1_search_1card
python search.py --epochs 1 \
--use_multiprocess False \
--batch_size 32 --use_gpu False >${log_path}/${model} 2>&1
print_info $? ${model}
#train
model=pcdarts_train_1card
python train.py --arch='PC_DARTS' \
--epochs 1 --use_multiprocess False \
--batch_size 32 --use_gpu False >${log_path}/${model} 2>&1
print_info $? ${model}
}


slimfacenet(){
cd ${slim_dir}/demo/slimfacenet || catchException slimfacenet
ln -s ${data_path}/slim/slimfacenet/CASIA CASIA
ln -s ${data_path}/slim/slimfacenet/lfw lfw
model=slim_slimfacenet_B75_train
CUDA_VISIBLE_DEVICES=${cudaid1} python -u train_eval.py \
--train_data_dir=./CASIA/ --test_data_dir=./lfw/ \
--action train --model=SlimFaceNet_B_x0_75 \
--start_epoch 0 --total_epoch 1 >${log_path}/slim_slimfacenet_B75_train 2>&1
print_info $? ${model}
model=slim_slimfacenet_B75_quan
CUDA_VISIBLE_DEVICES=${cudaid1} python train_eval.py \
--action quant --train_data_dir=./CASIA/ \
--test_data_dir=./lfw/  >${log_path}/slim_slimfacenet_B75_quan 2>&1
print_info $? ${model}
model=slim_slimfacenet_B75_eval
CUDA_VISIBLE_DEVICES=${cudaid1} python train_eval.py \
--action test --train_data_dir=./CASIA/ \
--test_data_dir=./lfw/ >${log_path}/slim_slimfacenet_B75_eval 2>&1
print_info $? ${model}
}

all_darts(){  # 2个模型
darts_1
#slimfacenet
}

####################################
#export all_case_list=(all_distillation all_quant all_prune all_nas all_darts)
export all_case_list=(all_distillation all_quant all_prune all_nas)

####################################
echo -e "\033[35m ---- start run case  \033[0m"
case_num=1
for model in ${all_case_list[*]};do
    echo -e "\033[35m ---- running $case_num/${#all_case_list[*]}: ${model}  \033[0m"
    ${model}
    let case_num++
done
echo -e "\033[35m ---- end run case  \033[0m"

cd ${slim_dir}/logs
FF=`ls *F*|wc -l`
if [ "${FF}" -gt "0" ];then
    exit 1
else
    exit 0
fi
