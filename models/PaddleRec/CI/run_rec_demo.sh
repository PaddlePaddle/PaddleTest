#! /bin/bash
# $1:模型类别 $2:模型名称
# $3:动态/静态-训练/推理（st_train、st_infer、dy_train、dy_infer）
# $4:是否用GPU $5:py文件路径 $6:config.yaml 路径
run_movie_recommand_case_func(){
    python -u $5 -m $6/config.yaml -o runner.use_gpu=$4  > ${log_path}/$1_$2_$3_$6_gpu_$4 2>&1
    print_info $? $1_$2_$3_$6_gpu_$4
}

demo_movie_recommand(){
declare -A dic
dic=([movie_recommand]='models/demo/movie_recommand' \
)

for model in $(echo ${!dic[*]});do
    model_path=${dic[$model]}
    echo ${model} : ${model_path}
    cd ${rec_dir}/${model_path}
    python -m pip instal pip install py27hash
    bash data_prepare.sh > ${log_path}/movie_recommand_data_prepare 2>&1
    print_info $? movie_recommand_data_prepare

    model_kind=`echo ${model_path} | awk -F '/' '{print $2}'`
    run_movie_recommand_case_func ${model_kind}  ${model} dy_train $1 "../../../tools/trainer.py" recall
    run_movie_recommand_case_func ${model_kind}  ${model} st_train $1 "../../../tools/static_trainer.py" recall
    run_movie_recommand_case_func ${model_kind}  ${model} dy_train $1 "../../../tools/trainer.py" rank
    run_movie_recommand_case_func ${model_kind}  ${model} st_train $1 "../../../tools/static_trainer.py" rank

    run_movie_recommand_case_func ${model_kind}  ${model} dy_infer $1 "../../../tools/infer.py" recall
    run_movie_recommand_case_func ${model_kind}  ${model} st_infer $1 "../../../tools/static_infer.py" recall
    run_movie_recommand_case_func ${model_kind}  ${model} dy_infer $1 "../../../tools/infer.py" rank
    run_movie_recommand_case_func ${model_kind}  ${model} st_infer $1 "../../../tools/static_infer.py" rank

    python parse.py recall_offline recall_infer_result > ${log_path}/movie_recommand_recall_offline_parse 2>&1
    print_info $? movie_recommand_recall_offline_parse
    python parse.py rank_offline rank_infer_result > ${log_path}/movie_recommand_rank_offline_parse 2>&1
    print_info $? movie_recommand_rank_offline_parse

done
}
