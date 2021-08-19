#!/bin/bash
####################################
export CUDA_VISIBLE_DEVICES=0
export all_data=/paddle/all_data/rec
#运行目录 PaddleRec/
export repo_path=$PWD
cd ${repo_path}
# git repo
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

###########
demo19(){
#必须先声明
declare -A dic
dic=([dnn]='models/rank/dnn' [wide_deep]='models/rank/wide_deep' [deepfm]='models/rank/deepfm' [fm]='models/rank/fm' [gateDnn]='models/rank/gateDnn' \
[logistic_regression]='models/rank/logistic_regression' [naml]='models/rank/naml' [ffm]='models/rank/ffm' [xdeepfm]='models/rank/xdeepfm' \
[esmm]='models/multitask/esmm' [mmoe]='models/multitask/mmoe' [ple]='models/multitask/ple' [share_bottom]='models/multitask/share_bottom' \
[dssm]='models/match/dssm' [match-pyramid]='models/match/match-pyramid' [multiview-simnet]='models/match/multiview-simnet' \
[tagspace]='models/contentunderstanding/tagspace' [textcnn]='models/contentunderstanding/textcnn' \
[ncf]='models/recall/ncf')
echo ${!dic[*]}   # 输出所有的key
echo ${dic[*]}    # 输出所有的value
i=1
for model in $(echo ${!dic[*]});do
    model_path=${dic[$model]}
    echo ${model} : ${model_path}
    cd ${repo_path}/${model_path}
    echo -e "\033[31m -------------$PWD---------------\n  \033[0m"
    # dygraph
    echo -e "\033[31m start dy train ${i} ${model}  \033[0m "
    python -u ../../../tools/trainer.py -m config.yaml -o runner.use_gpu=True
    print_info $? ${i}_${model}_dy_train
    echo -e "\033[31m start dy infer ${model}  \033[0m"
    python -u ../../../tools/infer.py -m config.yaml -o runner.use_gpu=True
    print_info $? ${i}_${model}_dy_infer
    rm -rf output_model_*

    # static
    echo -e "\033[31m start st train ${model}  \033[0m"
    python -u ../../../tools/static_trainer.py -m config.yaml -o runner.use_gpu=True
    print_info $? ${i}_${model}_st_train
    # 静态图预测
    echo -e "\033[31m start st infer ${model}  \033[0m"
    python -u ../../../tools/static_infer.py -m config.yaml -o runner.use_gpu=True
    print_info $? ${i}_${model}_st_infer
    let i+=1

done
}

word2vec(){
cd ${repo_path}/models/recall/word2vec
echo -e "\033[31m -------------$PWD-------------  \033[0m"
model=demo_word2vec
yaml_mode=config
if [[ "$1" =~ "con" ]]; then
model=all_word2vec
yaml_mode=config_bigdata
fi
# dygraph
echo -e "\033[31m start dy train 16 ${model} \n \033[0m "
python -u ../../../tools/trainer.py -m ${yaml_mode}.yaml -o runner.use_gpu=True
print_info $? ${model}_dy_train

echo -e "\033[31m start dy infer 16 ${model} \n \033[0m "
python -u infer.py -m ${yaml_mode}.yaml
print_info $? ${model}_dy_infer

rm -rf output_model_*

# 静态图训练
echo -e "\033[31m start st train 16 ${model} \n \033[0m "
python -u ../../../tools/static_trainer.py -m ${yaml_mode}.yaml -o runner.use_gpu=True
print_info $? ${model}_st_train

# 静态图预测
echo -e "\033[31m start st infer 16 ${model} \n \033[0m "
python -u static_infer.py -m ${yaml_mode}.yaml
print_info $? ${model}_st_infer

}

recall_demo(){
cd ${repo_path}/models/recall/$1
echo -e "\033[31m -------------$PWD-------------  \033[0m"
model=demo_$1
yaml_mode=config
if [[ "$2" =~ "con" ]]; then
model=all_$1
yaml_mode=config_bigdata
fi
# dygraph
echo -e "\033[31m start dy train 20 ${model} \n \033[0m "
python -u ../../../tools/trainer.py -m ${yaml_mode}.yaml -o runner.use_gpu=True
print_info $? ${model}_dy_train

echo -e "\033[31m start dy infer 20 ${model} \n \033[0m "
python -u infer.py -m ${yaml_mode}.yaml
print_info $? ${model}_dy_infer

rm -rf output_model_*

# 静态图训练
echo -e "\033[31m start st train 20 ${model} \n \033[0m "
python -u ../../../tools/static_trainer.py -m ${yaml_mode}.yaml -o runner.use_gpu=True
print_info $? ${model}_st_train

# 静态图预测
echo -e "\033[31m start st infer 20 ${model} \n \033[0m "
python -u static_infer.py -m ${yaml_mode}.yaml
print_info $? ${model}_st_infer

}
con_movie_recommand(){
cd ${repo_path}/models/demo/movie_recommand
echo -e "\033[31m $PWD  \033[0m"
python -m pip install py27hash
# download
bash data_prepare.sh

model=demo_movie_recommand_rank
# 动态图训练
python -u ../../../tools/trainer.py -m rank/config.yaml
# 动态图预测
python -u infer.py -m rank/config.yaml
# rank模型的测试结果解析
python parse.py rank_offline rank_infer_result
# 静态图训练
python -u ../../../tools/static_trainer.py -m rank/config.yaml
# 静态图预测
python -u static_infer.py -m rank/config.yaml
# recall模型的测试结果解析
python parse.py recall_offline recall_infer_result

model=demo_movie_recommand_recall
# 动态图训练
python -u ../../../tools/trainer.py -m recall/config.yaml
# 动态图预测
python -u infer.py -m recall/config.yaml
# rank模型的测试结果解析
python parse.py rank_offline rank_infer_result
# 静态图训练
python -u ../../../tools/static_trainer.py -m recall/config.yaml
# 静态图预测
python -u static_infer.py -m recall/config.yaml
# recall模型的测试结果解析
python parse.py recall_offline recall_infer_result
}

wide_deep_all(){
    cd ${repo_path}/models/rank/wide_deep
    model=demo_wide_deep_all
    # dy_cpu
    echo -e "\033[31m start _dy_train_cpu demo_wide_deep_all \033[0m "
    python -u ../../../tools/trainer.py -m config.yaml -o runner.use_gpu=False
    print_info $? ${model}_dy_train_cpu
    echo -e "\033[31m start _dy_infer_cpu demo_wide_deep_all \033[0m "
    python -u ../../../tools/infer.py -m config.yaml -o runner.use_gpu=False
    print_info $? ${model}_dy_infer_cpu
    rm -rf output

    # st_cpu
    echo -e "\033[31m start _st_train_cpu demo_wide_deep_all \033[0m "
    python -u ../../../tools/static_trainer.py -m config.yaml -o runner.use_gpu=False
    print_info $? ${model}_st_train_cpu
    echo -e "\033[31m start _st_infer_gpu1 demo_wide_deep_all \033[0m "
    python -u ../../../tools/static_infer.py -m config.yaml -o runner.use_gpu=False
    print_info $? ${model}_st_infer_cpu
    rm -rf output

    # dy_gpu1
    echo -e "\033[31m start _dy_train_gpu1 demo_wide_deep_all \033[0m "
    python -u ../../../tools/trainer.py -m config.yaml -o runner.use_gpu=True
    print_info $? ${model}_dy_train_gpu1
    echo -e "\033[31m start _dy_infer_gpu1 demo_wide_deep_all \033[0m "
    python -u ../../../tools/infer.py -m config.yaml -o runner.use_gpu=True
    print_info $? ${model}_dy_infer_gpu1
    rm -rf output

    # st_gpu1
    echo -e "\033[31m start _st_train_gpu1 demo_wide_deep_all \033[0m "
    python -u ../../../tools/static_trainer.py -m config.yaml -o runner.use_gpu=True
    print_info $? ${model}_st_train_gpu1
    echo -e "\033[31m start _st_infer_gpu1 demo_wide_deep_all \033[0m "
    python -u ../../../tools/static_infer.py -m config.yaml -o runner.use_gpu=True
    print_info $? ${model}_st_infer_gpu1
    rm -rf output

    # dy_gpu2
    echo -e "\033[31m start _dy_train_gpu2 dnn_all \033[0m "
    # sed -i '/runner:/a\  use_fleet: True' config.yaml
    fleetrun ../../../tools/trainer.py -m config.yaml -o runner.use_gpu=True runner.use_fleet=true
    print_info $? ${model}_dy_train_gpu2
    mv log ${model}_dy_train_gpu2_dist_logs
    echo -e "\033[31m start _dy_infer_gpu2 dnn_all \033[0m "
    fleetrun ../../../tools/infer.py -m config.yaml -o runner.use_gpu=True runner.use_fleet=true
    print_info $? ${model}_dy_infer_gpu2
    mv log ${model}_dy_infer_gpu2_dist_logs
    rm -rf output

    # st_gpu2
    echo -e "\033[31m start _st_train_gpu2 dnn_all \033[0m "
#    sed -i '/runner:/a\  use_fleet: True' config.yaml
    fleetrun ../../../tools/static_trainer.py -m config.yaml -o runner.use_gpu=True runner.use_fleet=true
    print_info $? ${model}_st_train_gpu2
    mv log ${model}_st_train_gpu2_dist_logs
    echo -e "\033[31m start _st_infer_gpu2 dnn_all \033[0m "
    fleetrun ../../../tools/static_infer.py -m config.yaml -o runner.use_gpu=True runner.use_fleet=true
    print_info $? ${model}_st_infer_gpu2
    mv log ${model}_st_infer_gpu2_dist_logs

}
################################################

download_all_data(){
echo  "start download  data"
cp -r ${repo_path}/datasets ${all_data}/
cd ${all_data}/datasets
for name in `ls`;
do
    if [ -d "${name}" ];then
    cd ${all_data}/datasets/${name}
    sh run.sh >${log_path}/con_${name}_down_data 2>&1
    print_info $? con_${name}_down_data
    cd -
    fi
done
}
con_dy_train_infer(){
echo "start run $1 dygraph con"
# 动态图训练
python -u ../../../tools/trainer.py -m config_bigdata.yaml -o runner.use_gpu=True > ${log_path}/con_$1_train 2>&1
print_info $? con_$1_train
# 动态图预测
python -u ../../../tools/infer.py -m config_bigdata.yaml -o runner.use_gpu=True > ${log_path}/con_$1_infer 2>&1
print_info $? con_$1_infer

}
con_st_train_infer(){
echo "start run $1 static con"
# 静态图训练
python -u ../../../tools/static_trainer.py -m config_bigdata.yaml -o runner.use_gpu=True > ${log_path}/con_$1_train 2>&1
print_info $? con_$1_train
# 静态图预测
python -u ../../../tools/static_infer.py -m config_bigdata.yaml -o runner.use_gpu=True > ${log_path}/con_$1_infer 2>&1
print_info $? con_$1_infer
}

con_dy_train_gpu2_infer(){
echo "start run $1 dygraph con"
# 动态图训练
python -m paddle.distributed.launch ../../../tools/trainer.py -m config_bigdata.yaml -o runner.use_gpu=True runner.use_fleet=True > ${log_path}/con_$1_train 2>&1
print_info $? con_$1_train
# 动态图预测
python -m paddle.distributed.launch ../../../tools/infer.py -m config_bigdata.yaml -o runner.use_gpu=True runner.use_fleet=True > ${log_path}/con_$1_infer 2>&1
print_info $? con_$1_infer

}
con_st_train_gpu2_infer(){
echo "start run $1 static con"
# 静态图训练
python -u ../../../tools/static_trainer.py -m config_bigdata.yaml -o runner.use_gpu=True runner.use_fleet=True > ${log_path}/con_$1_train_gpu2 2>&1
print_info $? con_$1_train_gpu2
# 静态图预测
python -u ../../../tools/static_infer.py -m config_bigdata.yaml -o runner.use_gpu=True runner.use_fleet=True > ${log_path}/con_$1_infer_gpu2 2>&1
print_info $? con_$1_infer_gpu2
}

con19(){
#必须先声明
declare -A dic
dic=([dnn]='models/rank/dnn' [wide_deep]='models/rank/wide_deep' [deepfm]='models/rank/deepfm' [fm]='models/rank/fm' [gateDnn]='models/rank/gateDnn' \
[logistic_regression]='models/rank/logistic_regression' [naml]='models/rank/naml' [ffm]='models/rank/ffm' [xdeepfm]='models/rank/xdeepfm' \
[esmm]='models/multitask/esmm' [mmoe]='models/multitask/mmoe' [ple]='models/multitask/ple' [share_bottom]='models/multitask/share_bottom' \
[dssm]='models/match/dssm' [match-pyramid]='models/match/match-pyramid' [multiview-simnet]='models/match/multiview-simnet' \
[tagspace]='models/contentunderstanding/tagspace' [textcnn]='models/contentunderstanding/textcnn' \
[ncf]='models/recall/ncf')
echo ${!dic[*]}   # 输出所有的key
echo ${dic[*]}    # 输出所有的value
i=1
for model in $(echo ${!dic[*]});do
    model_path=${dic[$model]}
    echo ${model} : ${model_path}
    cd ${repo_path}/${model_path}
    con_dy_train_infer ${i}_rank_${model}_dy
    mv output_model_all_${model} output_model_all_${model}_dy

    con_dy_train_gpu2_infer ${i}_rank_${model}_dy
    mv output_model_all_${model} output_model_all_${model}_dy_gpu2


#    con_st_train_infer ${i}_rank_${model}_st
#    mv output_model_all_${model} output_model_all_${model}_st
    let i+=1
done
}
################################################
run_demo(){
mkdir ${repo_path}/demo_log
export log_path=${repo_path}/demo_log
demo19
recall_demo word2vec
recall_demo mind
wide_deep_all
}
################################################
run_con(){
cd ${repo_path}
mkdir ${repo_path}/con_log
export log_path=${repo_path}/con_log
if [ ! -d "${all_data}/datasets" ];then
    download_all_data
fi
mv ${repo_path}/datasets ${repo_path}/datasets_bk
ln -s ${all_data}/datasets ${repo_path}/datasets

con19
con_movie_recommand
#word2vec con
}
################################################
#run_demo
#run_con

################################################
$1 || True
echo -e "\033[31m -------------result:-------------  \033[0m"
cat ${repo_path}/result.log
exit ${exit_flag}
