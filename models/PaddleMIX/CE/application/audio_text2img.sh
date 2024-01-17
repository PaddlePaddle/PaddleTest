#!/bin/bash

python applications/Audio2Img/audio2img_imagebind.py \
    --model_name_or_path imagebind-1.2b/ \
    --stable_unclip_model_name_or_path stabilityai/stable-diffusion-2-1-unclip \
    --input_audio https://paddlenlp.bj.bcebos.com/models/community/paddlemix/audio-files/bird_audio.wav \
    --input_text 'A photo.'
