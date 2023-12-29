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
from ppdiffusers import LDMTextToImagePipeline
model_name_or_path = "./ldm_pipelines"
pipe = LDMTextToImagePipeline.from_pretrained(model_name_or_path)
prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, guidance_scale=7.5).images[0]
image.save("astronaut_rides_horse.png")
