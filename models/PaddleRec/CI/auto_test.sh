#! /bin/bash
#export rec_dir=
if [ -d "${rec_dir}/logs" ];then
    rm -rf ${rec_dir}/logs
fi
mkdir ${rec_dir}/logs
export log_path=${rec_dir}/logs

print_info(){
if [ $1 -ne 0 ];then
    mv ${log_path}/$2 ${log_path}/FAIL_$2.log
    echo ---${log_path}/FAIL_$2---
    echo "fail log as follow"
    cat  ${log_path}/FAIL_$2.log
else
    mv ${log_path}/$2 ${log_path}/SUCCESS_$2.log
    echo ---${log_path}/SUCCESS_$2---
fi
}

# $1:模型类别、$2:模型名称、$3:动态/静态-训练/推理（st_train、st_infer、dy_train、dy_infer）
# $4:是否用GPU、$5:py文件路径、$6 $7:附加参数
run_case_func(){
    python -u $5 -m config.yaml -o runner.use_gpu=$4 $6 $7> ${log_path}/$1_$2_$3_gpu_$4 2>&1
    print_info $? $1_$2_$3_gpu_$4
}

demo_run_func(){
    cd ${rec_dir}/models/$1/
    for model in `ls -d */`
    do
        #去掉/ 如textcnn/
        model=${model%?}
        echo "${model} running"
        cd ${model}
        if [ ${model} == "kim" ];then
            echo ---skip ${model}---
            # run_case_func $1 ${model} dy_train $2 trainer.py -o mode=train
            # run_case_func $1 ${model} dy_infer $2 trainer.py -o mode=train
        elif [ ${model} == "ensfm" ] || [ ${model} == "mhcn" ] ;then
            run_case_func $1 ${model} dy_train $2 ../../../tools/trainer.py
            #run_case_func $1 ${model} dy_infer $2 infer.py
        elif [ ${model} == "mind" ] || [ ${model} == "word2vec" ] ;then
            run_case_func $1 ${model} dy_train $2 ../../../tools/trainer.py
            #run_case_func $1 ${model} dy_infer $2 infer.py
            run_case_func $1 ${model} st_train $2 ../../../tools/static_trainer.py
            #run_case_func $1 ${model} st_infer $2 static_infer.py
        elif [ ${model} == "metaheac" ] ;then
            run_case_func $1 ${model} dy_train $2 ../../../tools/trainer.py
            run_case_func $1 ${model} dy_infer $2 ./infer.py
        elif [ ${model} == "autofis" ] ;then
            run_case_func $1 ${model} dy_train $2 trainer.py 
            run_case_func $1 ${model} dy_train $2 trainer.py -o stage=1
            run_case_func $1 ${model} dy_infer $2 ../../../tools/infer.py -o stage=1
        elif [ ${model} == "dataset" ] || [ ${model} == "slot_dnn" ];then
            echo ---skip ${model} ---
        elif [ ${model} == "deeprec" ] ;then
            run_case_func $1 ${model} dy_train $2 trainer.py
            run_case_func $1 ${model} dy_infer $2 infer.py
        elif [ ${model} == "aitm" ] || [ ${model} == "maml" ] || [ ${model} == "bert4rec" ] || [ ${model} == "dcn_v2" ] \
            || [ ${model} == "fat_deepffm" ] || [ ${model} == "fgcnn" ] || [ ${model} == "flen" ] || [ ${model} == "iprec" ] \
            || [ ${model} == "sign" ] ;then
            run_case_func $1 ${model} dy_train $2 ../../../tools/trainer.py
            run_case_func $1 ${model} dy_infer $2 ../../../tools/infer.py
        elif [ ${model} == "deepwalk" ] ;then
            cd multi_class
            run_case_func $1 ${model} st_train $2 ../../../../tools/static_trainer.py
            run_case_func $1 ${model} st_infer $2 ../../../../tools/static_infer.py
            cd ..
        else
            run_case_func $1 ${model} dy_train $2 ../../../tools/trainer.py
            run_case_func $1 ${model} dy_infer $2 ../../../tools/infer.py
            run_case_func $1 ${model} st_train $2 ../../../tools/static_trainer.py
            run_case_func $1 ${model} st_infer $2 ../../../tools/static_infer.py
        fi
        cd ..
    done
}

demo_list=(contentunderstanding match multitask rank recall)
for demo in ${demo_list[*]}
do
    demo_run_func ${demo} True
done