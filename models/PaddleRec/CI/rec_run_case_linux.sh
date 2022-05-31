#! /bin/bash

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

# $1:模型类别 $2:模型名称
# $3:动态/静态-训练/推理（st_train、st_infer、dy_train、dy_infer）
# $4:是否用GPU $5:py文件路径 $6:附加参数
run_case_func(){
    python -u $5 -m config.yaml -o runner.use_gpu=$4 $6 > ${log_path}/$1_$2_$3_gpu_$4 2>&1
    print_info $? $1_$2_$3_gpu_$4
}

# 回归st_train、st_infer、dy_train、dy_infer
run_dy_st_train_infer(){
    run_case_func $1  $2 dy_train $3 "../../../tools/trainer.py"
    run_case_func $1  $2 dy_infer $3 "../../../tools/infer.py"
    run_case_func $1  $2 st_train $3 "../../../tools/static_trainer.py"
    run_case_func $1  $2 st_infer $3 "../../../tools/static_infer.py"
}

# $1:模型类别 $2:模型名称
# $3:动态/静态-训练/推理（st_train、st_infer、dy_train、dy_infer）
# $4:py文件路径 $5:附加参数
feetrun_case_func(){
    fleetrun -u $4 -m config.yaml -o runner.use_gpu=True runner.use_fleet=true $5 > ${log_path}/$1_$2_$3_fleet 2>&1
    print_info $? $1_$2_$3_fleet
}

feetrun_dy_st_train_infer(){
    feetrun_case_func $1 $2 dy_train "../../../tools/trainer.py"
    feetrun_case_func $1 $2 dy_infer "../../../tools/infer.py"
    feetrun_case_func $1 $2 st_train "../../../tools/static_trainer.py"
    feetrun_case_func $1 $2 st_infer "../../../tools/static_infer.py"
}

demo_contentunderstanding(){
declare -A dic
dic=([tagspace]='models/contentunderstanding/tagspace' [textcnn]='models/contentunderstanding/textcnn'\
)

for model in $(echo ${!dic[*]});do
    model_path=${dic[$model]}
    echo ${model} : ${model_path}
    cd ${rec_dir}/${model_path}
    model_kind=`echo ${model_path} | awk -F '/' '{print $2}'`
    # 回归st_train、st_infer、dy_train、dy_infer
    run_dy_st_train_infer ${model_kind} ${model} $1
    #fleetrun方式回归st_train、st_infer、dy_train、dy_infer
    if [ "$2" == "freet_run" ];then
        feetrun_dy_st_train_infer ${model_kind} ${model}
    fi
    rm -rf output*
done
}

demo_match(){
declare -A dic
dic=([dssm]='models/match/dssm' [match-pyramid]='models/match/match-pyramid'\
    [multiview-simnet]='models/match/multiview-simnet' \
)

for model in $(echo ${!dic[*]});do
    model_path=${dic[$model]}
    echo ${model} : ${model_path}
    cd ${rec_dir}/${model_path}
    model_kind=`echo ${model_path} | awk -F '/' '{print $2}'`
    run_dy_st_train_infer ${model_kind} ${model} $1

    if [ "$2" == "freet_run" ];then
        feetrun_dy_st_train_infer ${model_kind} ${model}
    fi
    rm -rf output*
done
}

demo_multitask(){
declare -A dic
dic=([aitm]='models/multitask/aitm' [dselect_k]='models/multitask/dselect_k' \
    [esmm]='models/multitask/esmm' [maml]='models/multitask/maml' [mmoe]='models/multitask/mmoe' \
    [ple]='models/multitask/ple' [share_bottom]='models/multitask/share_bottom' \
)

for model in $(echo ${!dic[*]});do
    model_path=${dic[$model]}
    echo ${model} : ${model_path}
    cd ${rec_dir}/${model_path}
    model_kind=`echo ${model_path} | awk -F '/' '{print $2}'`
    if [ ${model} == "aitm" ] || [ ${model} == "maml" ] || [ ${model} == "fat_deepffm" ] || [ ${model} == "fat_deepffm" ];then
        run_case_func ${model_kind} ${model} dy_train $1 "../../../tools/trainer.py"
        run_case_func ${model_kind} ${model} dy_infer $1 "../../../tools/infer.py"
    else
        run_dy_st_train_infer ${model_kind} ${model} $1

        if [ "$2" == "freet_run" ];then
            feetrun_dy_st_train_infer ${model_kind} ${model}
        fi
    fi
    rm -rf output*
done
}

demo_rank(){
declare -A dic
dic=([autofis]='models/rank/autofis' [bert4rec]='models/rank/bert4rec' [bst]='models/rank/bst' [dcn]='models/rank/dcn' [dcn_v2]='models/rank/dcn_v2' \
    [deepfefm]='models/rank/deepfefm' [deepfm]='models/rank/deepfm' [deeprec]='models/rank/deeprec' [dien]='models/rank/dien' \
    [difm]='models/rank/difm' [din]='models/rank/din' [dlrm]='models/rank/dlrm' [dmr]='models/rank/dmr' [dnn]='models/rank/dnn' \
    [dsin]='models/rank/dsin' [fat_deepffm]='models/rank/fat_deepffm' [ffm]='models/rank/ffm' [flen]='models/rank/flen' \
    [fm]='models/rank/fm' [gatenet]='models/rank/gatenet' [logistic_regression]='models/rank/logistic_regression' \
    [naml]='models/rank/naml' [wide_deep]='models/rank/wide_deep' [xdeepfm]='models/rank/xdeepfm' \
)

for model in $(echo ${!dic[*]});do
    model_path=${dic[$model]}
    echo ${model} : ${model_path}
    cd ${rec_dir}/${model_path}
    model_kind=`echo ${model_path} | awk -F '/' '{print $2}'`
    if [ ${model} == "autofis" ];then
        run_case_func ${model_kind}  ${model} dy_train $1 "trainer.py"
        run_case_func ${model_kind}  ${model} dy_train $1 "trainer.py" "stage=1"
        run_case_func ${model_kind}  ${model} dy_infer $1 "../../../tools/infer.py" "stage=1"
    elif [ ${model} == "deeprec" ];then
        run_case_func ${model_kind}  ${model} dy_train $1 "trainer.py"
        run_case_func ${model_kind}  ${model} dy_infer $1 "infer.py"
    elif [ ${model} == "bert4rec" ] || [ ${model} == "dcn_v2" ] || [ ${model} == "fat_deepffm" ] || [ ${model} == "fat_deepffm" ] || [ ${model} == "flen" ];then
        run_case_func ${model_kind}  ${model} dy_train $1 "../../../tools/trainer.py"
        run_case_func ${model_kind}  ${model} dy_infer $1 "../../../tools/infer.py"
    else
        run_dy_st_train_infer ${model_kind} ${model} $1

        if [ "$2" == "freet_run" ];then
            feetrun_dy_st_train_infer ${model_kind} ${model}
        fi
    fi
    rm -rf output*
done
}

demo_recall(){

declare -A dic
dic=([deepwalk]='models/recall/deepwalk' [ensfm]='models/recall/ensfm' \
    [mhcn]='models/recall/mhcn' [ncf]='models/recall/ncf' [tisas]='models/recall/tisas' \
    [mind]='models/recall/mind' [word2vec]='models/recall/word2vec' \
)

for model in $(echo ${!dic[*]});do
    model_path=${dic[$model]}
    echo ${model} : ${model_path}
    cd ${rec_dir}/${model_path}
    model_kind=`echo ${model_path} | awk -F '/' '{print $2}'`
    if [ ${model} == "ensfm" ] || [ ${model} == "tisas" ];then
        run_case_func ${model_kind}  ${model} dy_train $1 "../../../tools/trainer.py"
        run_case_func ${model_kind}  ${model} dy_infer $1 "infer.py"
    elif [ ${model} == "mhcn" ] ;then
         run_case_func ${model_kind}  ${model} dy_train $1 "../../../tools/trainer.py"
         # 需 解决：error: unrecognized arguments: -o runner.use_gpu=True
         # run_case_func ${model_kind}  ${model} dy_infer $1 "infer.py"
    elif [ ${model} == "deepwalk" ] ;then
        cd multi_class
        run_case_func ${model_kind} ${model} st_train $1 "../../../../tools/static_trainer.py"
        run_case_func ${model_kind} ${model} st_infer $1 "../../../../tools/static_infer.py"
    elif [ ${model} == "word2vec" ] || [ ${model} == "mind" ];then
        run_case_func ${model_kind} ${model} dy_train $1 "../../../tools/trainer.py"
        # run_case_func ${model_kind} ${model} dy_infer $1 "infer.py"
        run_case_func ${model_kind} ${model} st_train $1 "../../../tools/static_trainer.py"
        # run_case_func ${model_kind} ${model} st_infer $1 "static_infer.py"
    else
        run_dy_st_train_infer ${model_kind} ${model} $1

        if [ "$2" == "freet_run" ];then
            feetrun_dy_st_train_infer ${model_kind} ${model}
        fi
    fi
    rm -rf output*
done
}

demo_rerank(){
declare -A dic
dic=([rerank]='models/rerank/xxx' \
)

for model in $(echo ${!dic[*]});do
    model_path=${dic[$model]}
    echo ${model} : ${model_path}
    cd ${rec_dir}/${model_path}
    model_kind=`echo ${model_path} | awk -F '/' '{print $2}'`
    run_case_dy_func ${model_kind} ${model} $1
    run_case_st_func ${model_kind} ${model} $1
done
}

# rerank 暂时无模型
run_CI_func(){
    demo_contentunderstanding True
    demo_match True
    demo_multitask True
    demo_rank True
    demo_recall True
}

run_freet_func(){
    demo_contentunderstanding True freet_run
    demo_match True freet_run
    demo_multitask True freet_run
    demo_rank True freet_run
    demo_recall True freet_run
}

run_CPU_func(){
    demo_contentunderstanding False

    demo_match False
    demo_multitask False
    demo_rank False
    demo_recall False
}

run_demo_func(){
    source ./run_rec_demo.sh
    demo_movie_recommand True
}

print_logs(){
cd ${log_path}
FF=`ls *FAIL*|wc -l`
if [ "${FF}" -gt "0" ];then
    echo ---fail case: ${FF}
    ls *FAIL*
    exit 1
else
    echo ---all case pass---
    exit 0
fi
}

case $1 in
"run_CI")
    run_CI_func
    print_logs
    ;;
"run_CE")
    run_freet_func
    ;;
"run_CPU")
    run_CPU_func
    ;;
"run_ALL")
    run_freet_func
    run_CPU_func
    ;;
"run_demo")
    run_demo_func
    ;;
esac
