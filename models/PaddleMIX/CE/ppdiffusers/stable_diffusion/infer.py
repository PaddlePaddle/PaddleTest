# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
infer
"""
from ppdiffusers import StableDiffusionPipeline, UNet2DConditionModel
# 加载上面我们训练好的unet权重
unet_model_name_or_path = "./laion400m_pretrain_output_trainer/unet"
unet = UNet2DConditionModel.from_pretrained(unet_model_name_or_path)
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", safety_checker=None, unet=unet)
prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, guidance_scale=7.5, width=256, height=256).images[0]
image.save("astronaut_rides_horse.png")
