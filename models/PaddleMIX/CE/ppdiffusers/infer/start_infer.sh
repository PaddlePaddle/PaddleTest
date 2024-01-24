#!/bin/bash

cur_path=$(pwd)
echo ${cur_path}

work_path=${root_path}/PaddleMIX/ppdiffusers/examples/inference/
echo ${work_path}

work_path2=${root_path}/PaddleMIX/ppdiffusers/
echo ${work_path}

log_dir=${root_path}/infer_log


if [ ! -d "$log_dir" ]; then
    mkdir -p "$log_dir"
fi

/bin/cp -rf ./* ${work_path}

cd ${work_path2}
export http_proxy=${proxy}
export https_proxy=${proxy}
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pip install pytest safetensors ftfy fastcore opencv-python einops parameterized requests-mock
unset http_proxy
unset https_proxy

cd ${work_path}
exit_code=0

# Text-to-Image Generation

echo "*******infer text_to_image_generation-alt_diffusion begin***********"
(python text_to_image_generation-alt_diffusion.py) 2>&1 | tee ${log_dir}/text_to_image_generation-alt_diffusion.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_image_generation-alt_diffusion run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_image_generation-alt_diffusion run fail" >>"${log_dir}/infer_res_res.log"
fi
echo "*******infer text_to_image_generation-alt_diffusion end***********"

# Image-to-Image Text-Guided Generation
echo "*******infer image_to_image_text_guided_generation-alt_diffusion begin***********"
(python image_to_image_text_guided_generation-alt_diffusion.py) 2>&1 | tee ${log_dir}/image_to_image_text_guided_generation-alt_diffusion.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer image_to_image_text_guided_generation-alt_diffusion run success" >>"${log_dir}/infer_res.log"
else
    echo "infer image_to_image_text_guided_generation-alt_diffusion run fail" >>"${log_dir}/infer_res_res.log"
fi
echo "*******infer image_to_image_text_guided_generation-alt_diffusion end***********"

# Unconditional Audio Generation
echo "*******infer unconditional_audio_generation-audio_diffusion begin***********"
(python unconditional_audio_generation-audio_diffusion.py) 2>&1 | tee ${log_dir}/unconditional_audio_generation-audio_diffusion.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer unconditional_audio_generation-audio_diffusion run success" >>"${log_dir}/infer_res.log"
else
    echo "infer unconditional_audio_generation-audio_diffusion run fail" >>"${log_dir}/infer_res_res.log"
fi
echo "*******infer unconditional_audio_generation-audio_diffusion end***********"

# Image-to-Image Text-Guided Generation
echo "*******infer image_to_image_text_guided_generation-controlnet begin***********"
(python image_to_image_text_guided_generation-controlnet.py) 2>&1 | tee ${log_dir}/image_to_image_text_guided_generation-controlnet.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer image_to_image_text_guided_generation-controlnet run success" >>"${log_dir}/infer_res.log"
else
    echo "infer image_to_image_text_guided_generation-controlnet run fail" >>"${log_dir}/infer_res_res.log"
fi
echo "*******infer image_to_image_text_guided_generation-controlnet end***********"

# Unconditional Audio Generation
echo "*******infer unconditional_audio_generation-dance_diffusion begin***********"
(python unconditional_audio_generation-dance_diffusion.py) 2>&1 | tee ${log_dir}/unconditional_audio_generation-dance_diffusion.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer unconditional_audio_generation-dance_diffusion run success" >>"${log_dir}/infer_res.log"
else
    echo "infer unconditional_audio_generation-dance_diffusion run fail" >>"${log_dir}/infer_res_res.log"
fi
echo "*******infer unconditional_audio_generation-dance_diffusion end***********"

# Unconditional Image Generation
echo "*******infer unconditional_image_generation-ddpm begin***********"
(python unconditional_image_generation-ddpm.py) 2>&1 | tee ${log_dir}/unconditional_image_generation-ddpm.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer unconditional_image_generation-ddpm run success" >>"${log_dir}/infer_res.log"
else
    echo "infer unconditional_image_generation-ddpm run fail" >>"${log_dir}/infer_res_res.log"
fi
echo "*******infer unconditional_image_generation-ddpm end***********"

# Unconditional Image Generation
echo "*******infer unconditional_image_generation-ddim begin***********"
(python unconditional_image_generation-ddim.py) 2>&1 | tee ${log_dir}/unconditional_image_generation-ddim.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer unconditional_image_generation-ddim run success" >>"${log_dir}/infer_res.log"
else
    echo "infer unconditional_image_generation-ddim run fail" >>"${log_dir}/infer_res_res.log"
fi
echo "*******infer unconditional_image_generation-ddim end***********"

# Text-to-Image Generation
echo "*******infer text_to_image_generation-latent_diffusion begin***********"
(python text_to_image_generation-latent_diffusion.py) 2>&1 | tee ${log_dir}/text_to_image_generation-latent_diffusion.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_image_generation-latent_diffusion run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_image_generation-latent_diffusion run fail" >>"${log_dir}/infer_res_res.log"
fi
echo "*******infer text_to_image_generation-latent_diffusion end***********"

# Super Superresolution
echo "*******infer super_resolution-latent_diffusion begin***********"
(python super_resolution-latent_diffusion.py) 2>&1 | tee ${log_dir}/super_resolution-latent_diffusion.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer super_resolution-latent_diffusion run success" >>"${log_dir}/infer_res.log"
else
    echo "infer super_resolution-latent_diffusion run fail" >>"${log_dir}/infer_res_res.log"
fi
echo "*******infer super_resolution-latent_diffusion end***********"

# Unconditional Image Generation
echo "*******infer unconditional_image_generation-latent_diffusion_uncond begin***********"
(python unconditional_image_generation-latent_diffusion_uncond.py) 2>&1 | tee ${log_dir}/unconditional_image_generation-latent_diffusion_uncond.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer unconditional_image_generation-latent_diffusion_uncond run success" >>"${log_dir}/infer_res.log"
else
    echo "infer unconditional_image_generation-latent_diffusion_uncond run fail" >>"${log_dir}/infer_res_res.log"
fi
echo "*******infer unconditional_image_generation-latent_diffusion_uncond end***********"

# Image-Guided Image Inpainting
echo "*******infer image_guided_image_inpainting-paint_by_example begin***********"
(python image_guided_image_inpainting-paint_by_example.py) 2>&1 | tee ${log_dir}/image_guided_image_inpainting-paint_by_example.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer image_guided_image_inpainting-paint_by_example run success" >>"${log_dir}/infer_res.log"
else
    echo "infer image_guided_image_inpainting-paint_by_example run fail" >>"${log_dir}/infer_res_res.log"
fi
echo "*******infer image_guided_image_inpainting-paint_by_example end***********"

# Unconditional Image Generation
echo "*******infer unconditional_image_generation-pndm begin***********"
(python unconditional_image_generation-pndm.py) 2>&1 | tee ${log_dir}/unconditional_image_generation-pndm.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer unconditional_image_generation-pndm run success" >>"${log_dir}/infer_res.log"
else
    echo "infer unconditional_image_generation-pndm run fail" >>"${log_dir}/infer_res_res.log"
fi
echo "*******infer unconditional_image_generation-pndm end***********"

# Image Inpainting
echo "*******infer image_inpainting-repaint begin***********"
(python image_inpainting-repaint.py) 2>&1 | tee ${log_dir}/image_inpainting-repaint.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer image_inpainting-repaint run success" >>"${log_dir}/infer_res.log"
else
    echo "infer image_inpainting-repaint run fail" >>"${log_dir}/infer_res_res.log"
fi
echo "*******infer image_inpainting-repaint end***********"

# Unconditional Image Generation
echo "*******infer unconditional_image_generation-score_sde_ve begin***********"
(python unconditional_image_generation-score_sde_ve.py) 2>&1 | tee ${log_dir}/unconditional_image_generation-score_sde_ve.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer unconditional_image_generation-score_sde_ve run success" >>"${log_dir}/infer_res.log"
else
    echo "infer unconditional_image_generation-score_sde_ve run fail" >>"${log_dir}/infer_res_res.log"
fi
echo "*******infer unconditional_image_generation-score_sde_ve end***********"

# Text-Guided Generation
echo "*******infer text_guided_generation-semantic_stable_diffusion begin***********"
(python text_guided_generation-semantic_stable_diffusion.py) 2>&1 | tee ${log_dir}/text_guided_generation-semantic_stable_diffusion.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_guided_generation-semantic_stable_diffusion run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_guided_generation-semantic_stable_diffusion run fail" >>"${log_dir}/infer_res_res.log"
fi
echo "*******infer text_guided_generation-semantic_stable_diffusion end***********"

# Text-to-Image Generation
echo "*******infer text_to_image_generation-stable_diffusion begin***********"
(python text_to_image_generation-stable_diffusion.py) 2>&1 | tee ${log_dir}/text_to_image_generation-stable_diffusion.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_image_generation-stable_diffusion run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_image_generation-stable_diffusion run fail" >>"${log_dir}/infer_res_res.log"
fi
echo "*******infer text_to_image_generation-stable_diffusion end***********"

# Image-to-Image Text-Guided Generation
echo "*******infer image_to_image_text_guided_generation-stable_diffusion begin***********"
(python image_to_image_text_guided_generation-stable_diffusion.py) 2>&1 | tee ${log_dir}/image_to_image_text_guided_generation-stable_diffusion.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer image_to_image_text_guided_generation-stable_diffusion run success" >>"${log_dir}/infer_res.log"
else
    echo "infer image_to_image_text_guided_generation-stable_diffusion run fail" >>"${log_dir}/infer_res_res.log"
fi
echo "*******infer image_to_image_text_guided_generation-stable_diffusion end***********"

# Text-Guided Image Inpainting
echo "*******infer text_guided_image_inpainting-stable_diffusion begin***********"
(python text_guided_image_inpainting-stable_diffusion.py) 2>&1 | tee ${log_dir}/text_guided_image_inpainting-stable_diffusion.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_guided_image_inpainting-stable_diffusion run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_guided_image_inpainting-stable_diffusion run fail" >>"${log_dir}/infer_res_res.log"
fi
echo "*******infer text_guided_image_inpainting-stable_diffusion end***********"

# Text-to-Image Generation
echo "*******infer text_to_image_generation-stable_diffusion_2 begin***********"
(python text_to_image_generation-stable_diffusion_2.py) 2>&1 | tee ${log_dir}/text_to_image_generation-stable_diffusion_2.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_image_generation-stable_diffusion_2 run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_image_generation-stable_diffusion_2 run fail" >>"${log_dir}/infer_res_res.log"
fi
echo "*******infer text_to_image_generation-stable_diffusion_2 end***********"

# mage-to-Image Text-Guided Generation
echo "*******infer image_to_image_text_guided_generation-stable_diffusion_2 begin***********"
(python image_to_image_text_guided_generation-stable_diffusion_2.py) 2>&1 | tee ${log_dir}/image_to_image_text_guided_generation-stable_diffusion_2.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer image_to_image_text_guided_generation-stable_diffusion_2 run success" >>"${log_dir}/infer_res.log"
else
    echo "infer image_to_image_text_guided_generation-stable_diffusion_2 run fail" >>"${log_dir}/infer_res_res.log"
fi
echo "*******infer image_to_image_text_guided_generation-stable_diffusion_2 end***********"

# Text-Guided Image Inpainting
echo "*******infer text_guided_image_inpainting-stable_diffusion_2 begin***********"
(python text_guided_image_inpainting-stable_diffusion_2.py) 2>&1 | tee ${log_dir}/text_guided_image_inpainting-stable_diffusion_2.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_guided_image_inpainting-stable_diffusion_2 run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_guided_image_inpainting-stable_diffusion_2 run fail" >>"${log_dir}/infer_res_res.log"
fi
echo "*******infer text_guided_image_inpainting-stable_diffusion_2 end***********"

# Text-Guided Image Upscaling
echo "*******infer text_guided_image_upscaling-stable_diffusion_2 begin***********"
(python text_guided_image_upscaling-stable_diffusion_2.py) 2>&1 | tee ${log_dir}/text_guided_image_upscaling-stable_diffusion_2.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_guided_image_upscaling-stable_diffusion_2 run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_guided_image_upscaling-stable_diffusion_2 run fail" >>"${log_dir}/infer_res_res.log"
fi
echo "*******infer text_guided_image_upscaling-stable_diffusion_2 end***********"

# Text-to-Image Generation
echo "*******infer text_to_image_generation-stable_diffusion_safe begin***********"
(python text_to_image_generation-stable_diffusion_safe.py) 2>&1 | tee ${log_dir}/text_to_image_generation-stable_diffusion_safe.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_image_generation-stable_diffusion_safe run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_image_generation-stable_diffusion_safe run fail" >>"${log_dir}/infer_res_res.log"
fi
echo "*******infer text_to_image_generation-stable_diffusion_safe end***********"

# Unconditional Image Generation
echo "*******infer unconditional_image_generation-stochastic_karras_ve begin***********"
(python unconditional_image_generation-stochastic_karras_ve.py) 2>&1 | tee ${log_dir}/unconditional_image_generation-stochastic_karras_ve.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer unconditional_image_generation-stochastic_karras_ve run success" >>"${log_dir}/infer_res.log"
else
    echo "infer unconditional_image_generation-stochastic_karras_ve run fail" >>"${log_dir}/infer_res_res.log"
fi
echo "*******infer unconditional_image_generation-stochastic_karras_ve end***********"

# Text-to-Image Generation
echo "*******infer text_to_image_generation-unclip begin***********"
(python python text_to_image_generation-unclip.py) 2>&1 | tee ${log_dir}/text_to_image_generation-unclip.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_image_generation-unclip run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_image_generation-unclip run fail" >>"${log_dir}/infer_res_res.log"
fi
echo "*******infer text_to_image_generation-unclip end***********"

# Text-to-Image Generation
echo "*******infer text_to_image_generation-versatile_diffusion begin***********"
(python text_to_image_generation-versatile_diffusion.py) 2>&1 | tee ${log_dir}/text_to_image_generation-versatile_diffusion.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_image_generation-versatile_diffusion run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_image_generation-versatile_diffusion run fail" >>"${log_dir}/infer_res_res.log"
fi
echo "*******infer text_to_image_generation-versatile_diffusion end***********"

# Image Variation
echo "*******infer image_variation-versatile_diffusion begin***********"
(python image_variation-versatile_diffusion.py) 2>&1 | tee ${log_dir}/image_variation-versatile_diffusion.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer image_variation-versatile_diffusion run success" >>"${log_dir}/infer_res.log"
else
    echo "infer image_variation-versatile_diffusion run fail" >>"${log_dir}/infer_res_res.log"
fi
echo "*******infer image_variation-versatile_diffusion end***********"

# Dual Text and Image Guided Generation
echo "*******infer dual_text_and_image_guided_generation-versatile_diffusion begin***********"
(python dual_text_and_image_guided_generation-versatile_diffusion.py) 2>&1 | tee ${log_dir}/dual_text_and_image_guided_generation-versatile_diffusion.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer dual_text_and_image_guided_generation-versatile_diffusion run success" >>"${log_dir}/infer_res.log"
else
    echo "infer dual_text_and_image_guided_generation-versatile_diffusion run fail" >>"${log_dir}/infer_res_res.log"
fi
echo "*******infer dual_text_and_image_guided_generation-versatile_diffusion end***********"

# Text-to-Image Generation
echo "*******infer text_to_image_generation-vq_diffusion begin***********"
(python text_to_image_generation-vq_diffusion.py) 2>&1 | tee ${log_dir}/text_to_image_generation-vq_diffusion.log
tmp_exit_code=${PIPESTATUS[0]}
exit_code=$(($exit_code + ${tmp_exit_code}))
if [ ${tmp_exit_code} -eq 0 ]; then
    echo "infer text_to_image_generation-vq_diffusion run success" >>"${log_dir}/infer_res.log"
else
    echo "infer text_to_image_generation-vq_diffusion run fail" >>"${log_dir}/infer_res_res.log"
fi
echo "*******infer text_to_image_generation-vq_diffusion end***********"
