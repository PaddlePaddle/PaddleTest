#!/bin/bash

echo "*******paddlemix imagebind predict begin***********"

(python run_predict.py \
--model_name_or_path imagebind-1.2b/ \
--input_text "A dog." \
--input_image https://paddlenlp.bj.bcebos.com/models/community/paddlemix/audio-files/dog_image.jpg \
--input_audio https://paddlenlp.bj.bcebos.com/models/community/paddlemix/audio-files/wave.wav) 2>&1 | tee ${log_dir}/run_imagebind_predict.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "paddlemix imagebind predict run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "paddlemix imagebind predict run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******paddlemix imagebind predict end***********"

# 检查命令是否成功执行
if [ ${exit_code} -ne 0 ]; then
  exit 1
fi