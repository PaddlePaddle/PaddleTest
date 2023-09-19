#!/bin/bash

python applications/Audio2Img/audio2img_imagebind.py \
--model_name_or_path The dir name of imagebind checkpoint. \
--stable_unclip_model_name_or_path The dir name of StableUnCLIPImg2ImgPipeline pretrained checkpoint. \
--input_audio wave.wav \
--input_image dog_image.jpg