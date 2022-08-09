#!/bin/bash
####################################
#export CUDA_VISIBLE_DEVICES=2
#运行目录 PaddleRec/
export repo_path=$PWD
cd ${repo_path}
#git clone https://github.com/PaddlePaddle/PaddleRec.git -b master
python -m pip list
export exit_flag=0
print_info(){
if [ $1 -ne 0 ];then
    exit_flag=1
    echo -e "\033[31m FAIL_$2 \033[0m"
    echo -e "\033[31m FAIL_$2 \033[0m"  >>${repo_path}/result.log
else
    echo -e "\033[32m SUCCESS_$2 \033[0m"
    echo -e "\033[32m SUCCESS_$2 \033[0m"  >>${repo_path}/result.log
fi
}
#######################################
rank_demo(){
cd ${repo_path}/models/rank/$1
echo -e "\033[31m -------------$PWD-------------  \033[0m"
model=demo_$1
yaml_mode=config
# 动态图训练
echo -e "\033[31m start dy train ${model} \n \033[0m "
python -u ../../../tools/trainer.py -m ${yaml_mode}.yaml
print_info $? ${model}_dy_train
# 动态图预测
echo -e "\033[31m start dy infer ${model} \n \033[0m "
python -u ../../../tools/infer.py -m ${yaml_mode}.yaml
print_info $? ${model}_dy_infer
rm -rf output_model_*
# 静态图训练
echo -e "\033[31m start st train ${model} \n \033[0m "
python -u ../../../tools/static_trainer.py -m ${yaml_mode}.yaml
print_info $? ${model}_st_train
# 静态图预测
echo -e "\033[31m start st infer ${model} \n \033[0m "
python -u ../../../tools/static_infer.py -m ${yaml_mode}.yaml
print_info $? ${model}_st_infer
}

contentunderstanding_demo(){
cd ${repo_path}/models/contentunderstanding/$1
echo -e "\033[31m -------------$PWD-------------  \033[0m"
model=demo_$1
yaml_mode=config
# 动态图训练
echo -e "\033[31m start dy train ${model} \n \033[0m "
python -u ../../../tools/trainer.py -m ${yaml_mode}.yaml
print_info $? ${model}_dy_train
# 动态图预测
echo -e "\033[31m start dy infer ${model} \n \033[0m "
python -u ../../../tools/infer.py -m ${yaml_mode}.yaml
print_info $? ${model}_dy_infer
rm -rf output_model_*
# 静态图训练
echo -e "\033[31m start st train ${model} \n \033[0m "
python -u ../../../tools/static_trainer.py -m ${yaml_mode}.yaml
print_info $? ${model}_st_train
# 静态图预测
echo -e "\033[31m start st infer ${model} \n \033[0m "
python -u ../../../tools/static_infer.py -m ${yaml_mode}.yaml
print_info $? ${model}_st_infer
}

multitask_demo(){
cd ${repo_path}/models/multitask/$1
echo -e "\033[31m -------------$PWD-------------  \033[0m"
model=demo_$1
yaml_mode=config
# 动态图训练
echo -e "\033[31m start dy train ${model} \n \033[0m "
python -u ../../../tools/trainer.py -m ${yaml_mode}.yaml
print_info $? ${model}_dy_train
# 动态图预测
echo -e "\033[31m start dy infer ${model} \n \033[0m "
python -u ../../../tools/infer.py -m ${yaml_mode}.yaml
print_info $? ${model}_dy_infer
rm -rf output_model_*
# 静态图训练
echo -e "\033[31m start st train ${model} \n \033[0m "
python -u ../../../tools/static_trainer.py -m ${yaml_mode}.yaml
print_info $? ${model}_st_train
# 静态图预测
echo -e "\033[31m start st infer ${model} \n \033[0m "
python -u ../../../tools/static_infer.py -m ${yaml_mode}.yaml
print_info $? ${model}_st_infer
}

match_demo(){
cd ${repo_path}/models/match/$1
echo -e "\033[31m -------------$PWD-------------  \033[0m"
model=demo_$1
yaml_mode=config
# 动态图训练
echo -e "\033[31m start dy train ${model} \n \033[0m "
python -u ../../../tools/trainer.py -m ${yaml_mode}.yaml
print_info $? ${model}_dy_train
# 动态图预测
echo -e "\033[31m start dy infer ${model} \n \033[0m "
python -u ../../../tools/infer.py -m ${yaml_mode}.yaml
print_info $? ${model}_dy_infer
rm -rf output_model_*
# 静态图训练
echo -e "\033[31m start st train ${model} \n \033[0m "
python -u ../../../tools/static_trainer.py -m ${yaml_mode}.yaml
print_info $? ${model}_st_train
# 静态图预测
echo -e "\033[31m start st infer ${model} \n \033[0m "
python -u ../../../tools/static_infer.py -m ${yaml_mode}.yaml
print_info $? ${model}_st_infer
}

recall_demo(){
cd ${repo_path}/models/recall/$1
echo -e "\033[31m -------------$PWD-------------  \033[0m"
model=demo_$1
yaml_mode=config
# dygraph
echo -e "\033[31m start dy train ${model} \n \033[0m "
python -u ../../../tools/trainer.py -m ${yaml_mode}.yaml
print_info $? ${model}_dy_train
echo -e "\033[31m start dy infer ${model} \n \033[0m "
python -u infer.py -m ${yaml_mode}.yaml
print_info $? ${model}_dy_infer
rm -rf output_model_*
# 静态图训练
echo -e "\033[31m start st train ${model} \n \033[0m "
python -u ../../../tools/static_trainer.py -m ${yaml_mode}.yaml
print_info $? ${model}_st_train
# 静态图预测
echo -e "\033[31m start st infer ${model} \n \033[0m "
python -u static_infer.py -m ${yaml_mode}.yaml
print_info $? ${model}_st_infer
}

recall_ncf(){
cd ${repo_path}/models/recall/$1
echo -e "\033[31m -------------$PWD-------------  \033[0m"
model=demo_$1
yaml_mode=config
# 动态图训练
echo -e "\033[31m start dy train ${model} \n \033[0m "
python -u ../../../tools/trainer.py -m ${yaml_mode}.yaml
print_info $? ${model}_dy_train
# 动态图预测
echo -e "\033[31m start dy infer ${model} \n \033[0m "
python -u ../../../tools/infer.py -m ${yaml_mode}.yaml
print_info $? ${model}_dy_infer
rm -rf output_model_*
# 静态图训练
echo -e "\033[31m start st train ${model} \n \033[0m "
python -u ../../../tools/static_trainer.py -m ${yaml_mode}.yaml
print_info $? ${model}_st_train
# 静态图预测
echo -e "\033[31m start st infer ${model} \n \033[0m "
python -u ../../../tools/static_infer.py -m ${yaml_mode}.yaml
print_info $? ${model}_st_infer
}
################################################
run_demo(){
mkdir ${repo_path}/demo_log
export log_path=${repo_path}/demo_log
# rank
rank_demo dnn
rank_demo wide_deep
rank_demo deepfm
rank_demo fm
rank_demo gatenet
rank_demo logistic_regression
rank_demo naml
rank_demo ffm
rank_demo xdeepfm
rank_demo bst
rank_demo dcn
rank_demo deepfefm
rank_demo dien
rank_demo din
rank_demo dlrm
rank_demo dmr
rank_demo difm
# multitask
multitask_demo esmm
multitask_demo mmoe
multitask_demo ple
multitask_demo share_bottom
#match
match_demo match-pyramid
match_demo multiview-simnet
match_demo dssm
# contentunderstanding
contentunderstanding_demo tagspace
contentunderstanding_demo textcnn
# recall
recall_demo word2vec
recall_demo mind
recall_ncf ncf
}
################################################
run_demo
################################################
$1 || True
echo -e "\033[31m -------------result:-------------  \033[0m"
cat ${repo_path}/result.log
exit ${exit_flag}
