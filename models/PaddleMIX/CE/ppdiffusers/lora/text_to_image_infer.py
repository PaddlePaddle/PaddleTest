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
text_to_image_infer
"""
from ppdiffusers import StableDiffusionPipeline
import paddle

model_path = "./sd-pokemon-model-lora"
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", paddle_dtype=paddle.float32)
# 注意：如果我们想从 HF Hub 加载权重，那么我们需要设置 from_hf_hub=True
pipe.unet.load_attn_procs(model_path)

prompt = "Totoro"
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("Totoro.png")
