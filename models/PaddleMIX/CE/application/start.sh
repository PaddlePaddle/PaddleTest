#!/bin/bash

cur_path=`pwd`
echo ${cur_path}


work_path=${root_path}/PaddleMIX/
echo ${work_path}

log_dir=${root_path}/log

# 检查上一级目录中是否存在log目录
if [ ! -d "$log_dir" ]; then
    # 如果log目录不存在，则创建它
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}/
exit_code=0

# 下载依赖、nltk_data和数据
bash prepare.sh

cd ${work_path}

echo "*******application grounded_sam begin***********"
(python grounded_sam.py) 2>&1 | tee ${log_dir}/grounded_sam.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "application grounded_sam run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "application grounded_sam run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******application grounded_sam end***********"


echo "*******application automatic_label begin***********"
(python automatic_label.py) 2>&1 | tee ${log_dir}/automatic_label.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "application automatic_label run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "application automatic_label run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******application automatic_label end***********"

echo "*******application grounded_sam begin***********"
(python grounded_sam.py) 2>&1 | tee ${log_dir}/grounded_sam.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "application grounded_sam_inpainting run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "application grounded_sam_inpainting run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******application grounded_sam_inpainting end***********"

echo "*******application grounded_sam_inpainting begin***********"
(python grounded_sam_inpainting.py) 2>&1 | tee ${log_dir}/grounded_sam_inpainting.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "application grounded_sam_inpainting run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "application grounded_sam_inpainting run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******application grounded_sam_inpainting end***********"

echo "*******application grounded_sam_chatglm begin***********"
(python grounded_sam_chatglm.py) 2>&1 | tee ${log_dir}/grounded_sam_chatglm.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "application grounded_sam_chatglm run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "application grounded_sam_chatglm run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******application grounded_sam_chatglm end***********"

echo "*******application text_guided_image_inpainting begin***********"
(python text_guided_image_inpainting.py) 2>&1 | tee ${log_dir}/text_guided_image_inpainting.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "application text_guided_image_inpainting run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "application text_guided_image_inpainting run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******application text_guided_image_inpainting end***********"

echo "*******application text_to_image_generation begin***********"
(python text_to_image_generation.py) 2>&1 | tee ${log_dir}/text_to_image_generation.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "application text_to_image_generation run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "application text_to_image_generation run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******application text_to_image_generation end***********"

echo "*******application text_guided_image_upscaling begin***********"
(python text_guided_image_upscaling.py) 2>&1 | tee ${log_dir}/text_guided_image_upscaling.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "application text_guided_image_upscaling run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "application text_guided_image_upscaling run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******application text_guided_image_upscaling end***********"

# echo "*******application dual_text_image_guided_generation begin***********"
# (python dual_text_image_guided_generation.py) | tee ${log_dir}/dual_text_image_guided_generation.log
# tmp_exit_code=${PIPESTATUS[0]}
# exit_code=$(($exit_code + ${tmp_exit_code}))
# if [ ${tmp_exit_code} -eq 0 ]; then
#     # 如果返回状态为0（成功），则追加成功消息到ce_res.log
#     echo "application dual_text_image_guided_generation run success" >> "${log_dir}/ce_res.log"
# else
#     # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
#     echo "application dual_text_image_guided_generation run fail" >> "${log_dir}/ce_res.log"
# fi
# echo "*******application dual_text_image_guided_generation end***********"

echo "*******application image2image_text_guided_generation begin***********"
(python image2image_text_guided_generation.py) 2>&1 | tee ${log_dir}/image2image_text_guided_generation.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "application image2image_text_guided_generation run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "application image2image_text_guided_generation run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******application image2image_text_guided_generation end***********"

echo "*******application text2video_generation begin***********"
(python text2video_generation.py) 2>&1 | tee ${log_dir}/text2video_generation.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "application text2video_generation run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "application text2video_generation run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******application text2video_generation end***********"

echo "*******application audio2caption_generation begin***********"
(python audio2caption_generation.py) 2>&1 | tee ${log_dir}/audio2caption_generation.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "application audio2caption_generation run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "application audio2caption_generation run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******application audio2caption_generation end***********"

echo "*******application audio2chat_generation begin***********"
(python audio2chat_generation.py) 2>&1 | tee ${log_dir}/audio2chat_generation.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "application audio2chat_generation run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "application audio2chat_generation run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******application audio2chat_generation end***********"

echo "*******application music_generation begin***********"
(python music_generation.py) 2>&1 | tee ${log_dir}/music_generation.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "application music_generation run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "application music_generation run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******application music_generation end***********"

echo "*******application audio2img begin***********"
(bash audio2img.sh) 2>&1 | tee ${log_dir}/audio2img.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "application audio2img run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "application audio2img run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******application audio2img end***********"

echo "*******application audio_text2img begin***********"
(bash audio_text2img.sh) 2>&1 | tee ${log_dir}/audio_text2img.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "application audio_text2img run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "application audio_text2img run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******application audio_text2img end***********"

echo "*******application auodio_image2image begin***********"
(bash auodio_image2image.sh) 2>&1 | tee ${log_dir}/auodio_image2image.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    # 如果返回状态为0（成功），则追加成功消息到ce_res.log
    echo "application auodio_image2image run success" >> "${log_dir}/ce_res.log"
else
    # 如果返回状态不为0（失败），则追加失败消息到ce_res.log
    echo "application auodio_image2image run fail" >> "${log_dir}/ce_res.log"
fi
echo "*******application auodio_image2image end***********"

cat ${log_dir}/ce_res.log

echo exit_code:${exit_code}
exit $exit_code