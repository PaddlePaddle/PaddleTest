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
automatic_label
"""
from paddlemix.appflow import Appflow
from ppdiffusers.utils import load_image
task = Appflow(app="auto_label",
               models=["paddlemix/blip2-caption-opt2.7b", "GroundingDino/groundingdino-swint-ogc", "Sam/SamVitH-1024"])
url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"
image_pil = load_image(url)
blip2_prompt = 'describe the image'
result = task(image=image_pil, blip2_prompt=blip2_prompt)
