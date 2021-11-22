#!/bin/bash
# 外部传入参数
# $1 训练使用单卡or多卡

# export PYTHONPATH=`pwd`:$PYTHONPATH

#获取当前路径
cur_path=`pwd`
model=${PWD##*/}
echo "${model} 模型数train阶段"
#配置目标数据存储路径
root_path=$cur_path/../../
#创建日志路径
if [[ ! -d "$cur_path/../../log/${model}" ]];then
mkdir -p "$cur_path/../../log/${model}"
fi

print_info(){
if [ $1 -ne 0 ];then
    echo -e "${model},$2,FAIL"
    echo "exit_code: 1.0" >> ../log/${model}/${model}_$2.log
else
    echo -e "${model},$2,SUCCESS"
    echo "exit_code: 0.0" >> ../log/${model}/${model}_$2.log
fi
}

#train
train_fcn_convergence(){
    python -m paddle.distributed.launch train.py \
       --config configs/fcn/fcn_hrnetw18_cityscapes_1024x512_80k.yml \
       --do_eval  >../log/${model}/${model}_train_convergence.log 2>&1
    print_info $? train_convergence
    echo  "Iou: `grep image: ../log/fcn_hrnetw18_cityscapes_1024x512_80k/fcn_hrnetw18_cityscapes_1024x512_80k_train_convergence.log | tail -1 | awk '{print $NF}'`" >>../log/fcn_hrnetw18_cityscapes_1024x512_80k/fcn_hrnetw18_cityscapes_1024x512_80k_train_convergence.log
}


cd $cur_path/../../PaddleSeg
train_fcn_convergence
kill -9 `ps afx | grep train | awk '{print $1}'`
