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
image2image_text_guided_generation
"""
import paddle
from paddlemix.appflow import Appflow
from PIL import Image
from ppdiffusers.utils import load_image

url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/image_Kurisu.png"
image = load_image(url).resize((512, 768))
prompt = "a red car in the sun"

paddle.seed(42)
prompt = "Kurisu Makise, looking at viewer, long hair, standing, 1girl, hair ornament, hair flower, cute, jacket, white flower, white dress"
negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"


app = Appflow(app='image2image_text_guided_generation',
              models=['Linaqruf/anything-v3.0'])
image = app(prompt=prompt, negative_prompt=negative_prompt,
            image=image)['result']

image.save("image_Kurisu_img2img.png")
