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
from ppdiffusers import StableDiffusionAdapterPipeline, T2IAdapter
from ppdiffusers.utils import load_image
adapter = T2IAdapter.from_pretrained("./sd15_openpose/checkpoint-100/adapter")
pipe = StableDiffusionAdapterPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", adapter=adapter, safety_checker=None)
pose_image = load_image(
    "https://paddlenlp.bj.bcebos.com/models/community/westfish/t2i-adapter/test/man-openpose.png")
img = pipe(prompt="a beautiful girl", image=pose_image,
           guidance_scale=9, num_inference_steps=50).images[0]
img.save("demo.png")
