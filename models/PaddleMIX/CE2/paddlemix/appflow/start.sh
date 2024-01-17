#!/bin/bash

export PPNLP_HOME=/home/cache_weight
export PPMIX_HOME=/home/cache_weight

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/tests/appflow
echo ${work_path}

log_dir=${root_path}/log

if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}

cd ${work_path}

exit_code=0

echo "*******paddlemix appflow test_cviw begin***********"
(python test_cviw.py) 2>&1 | tee ${log_dir}/test_cviw.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix appflow test_cviw success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix appflow test_cviw fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix appflow test_cviw end***********"

echo "*******paddlemix appflow test_inpainting begin***********"
(python test_inpainting.py) 2>&1 | tee ${log_dir}/test_inpainting.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix appflow test_inpainting success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix appflow test_inpainting fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix appflow test_inpainting end***********"

export RUN_SLOW_TEST=True
echo "*******export RUN_SLOW_TEST=True paddlemix appflow test_cviw begin***********"
(python test_cviw.py) 2>&1 | tee ${log_dir}/test_cviw_2.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "export RUN_SLOW_TEST=True paddlemix appflow test_cviw success" >>"${log_dir}/ce_res.log"
else
    echo "export RUN_SLOW_TEST=True paddlemix appflow test_cviw fail" >>"${log_dir}/ce_res.log"
fi
echo "*******export RUN_SLOW_TEST=True paddlemix appflow test_cviw end***********"
unset RUN_SLOW_TEST

echo "*******paddlemix appflow test_DualTextAndImageGuidedGeneration begin***********"
(python test_DualTextAndImageGuidedGeneration.py) 2>&1 | tee ${log_dir}/test_DualTextAndImageGuidedGeneration.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix appflow test_DualTextAndImageGuidedGeneration success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix appflow test_DualTextAndImageGuidedGeneration fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix appflow test_DualTextAndImageGuidedGeneration end***********"

echo "*******paddlemix appflow test_Image2ImageTextGuidedGeneration begin***********"
(python test_Image2ImageTextGuidedGeneration.py) 2>&1 | tee ${log_dir}/test_Image2ImageTextGuidedGeneration.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix appflow test_Image2ImageTextGuidedGeneration success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix appflow test_Image2ImageTextGuidedGeneration fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix appflow test_Image2ImageTextGuidedGeneration end***********"

# echo "*******paddlemix appflow test_MusicGeneration begin***********"
# (python test_MusicGeneration.py) 2>&1 | tee ${log_dir}/test_MusicGeneration.log
# tmp_exit_code=${PIPESTATUS[0]}
# exit_code=$(($exit_code + ${tmp_exit_code}))
# if [ ${tmp_exit_code} -eq 0 ]; then
#     echo "paddlemix appflow test_MusicGeneration success" >> "${log_dir}/ce_res.log"
# else
#     echo "paddlemix appflow test_MusicGeneration fail" >> "${log_dir}/ce_res.log"
# fi
# echo "*******paddlemix appflow test_MusicGeneration end***********"

echo "*******paddlemix appflow test_TextGuidedImageInpainting begin***********"
(python test_TextGuidedImageInpainting.py) 2>&1 | tee ${log_dir}/test_TextGuidedImageInpainting.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix appflow test_TextGuidedImageInpainting success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix appflow test_TextGuidedImageInpainting fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix appflow test_TextGuidedImageInpainting end***********"

echo "*******paddlemix appflow test_TextGuidedImageUpscaling begin***********"
(python test_TextGuidedImageUpscaling.py) 2>&1 | tee ${log_dir}/test_TextGuidedImageUpscaling.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix appflow test_TextGuidedImageUpscaling success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix appflow test_TextGuidedImageUpscaling fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix appflow test_TextGuidedImageUpscaling end***********"

echo "*******paddlemix appflow test_audio-to-Caption begin***********"
(python test_audio-to-Caption.py) 2>&1 | tee ${log_dir}/test_audio-to-Caption.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix appflow test_audio-to-Caption success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix appflow test_audio-to-Caption fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix appflow test_audio-to-Caption end***********"

echo "*******paddlemix appflow test_audio_chat begin***********"
(python test_audio_chat.py) 2>&1 | tee ${log_dir}/test_audio_chat.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix appflow test_audio_chat success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix appflow test_audio_chat fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix appflow test_audio_chat end***********"

echo "*******paddlemix appflow test_autolabel begin***********"
(python test_autolabel.py) 2>&1 | tee ${log_dir}/test_autolabel.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix appflow test_autolabel success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix appflow test_autolabel fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix appflow test_autolabel end***********"

echo "*******paddlemix appflow test_text2image begin***********"
(python test_text2image.py) 2>&1 | tee ${log_dir}/test_text2image.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix appflow test_text2image success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix appflow test_text2image fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix appflow test_text2image end***********"

echo "*******paddlemix appflow test_text2video begin***********"
(python test_text2video.py) 2>&1 | tee ${log_dir}/test_text2video.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "paddlemix appflow test_text2video success" >>"${log_dir}/ce_res.log"
else
    echo "paddlemix appflow test_text2video fail" >>"${log_dir}/ce_res.log"
fi
echo "*******paddlemix appflow test_text2video end***********"

echo exit_code:${exit_code}
exit ${exit_code}
