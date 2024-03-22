#!/bin/bash

log_dir=${root_path}/log

exit_code=0

echo "*******paddlemix minigpt4_13b predict begin***********"

wget https://paddlenlp.bj.bcebos.com/models/community/minigpt4-13b/model_state.pdparams
wget https://paddlenlp.bj.bcebos.com/models/community/minigpt4-13b/minigpt4_13b.tar.gz
tar -zxvf minigpt4_13b.tar.gz
mv model_state.pdparams minigpt4_13b/
rm -rf minigpt4_13b.tar.gz

(python run_predict.py \
    --pretrained_name_or_path minigpt4_13b \
    --image_path "example.png" \
    --decode_strategy "greedy_search" \
    --max_length 300 \
    --num_beams 1 \
    --top_p 1.0 \
    --top_k 0 \
    --repetition_penalty 1.0 \
    --length_penalty 0.0 \
    --temperature 1.0) 2>&1 | tee ${log_dir}/run_minigpt4_13b_predict.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix minigpt4_13b predict run success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix minigpt4_13b predict run fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix minigpt4_13b predict end***********"

# 检查命令是否成功执行
if [ ${exit_code} -ne 0 ]; then
    exit 1
fi
