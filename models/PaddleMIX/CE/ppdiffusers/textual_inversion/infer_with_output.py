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
infer_with_output
"""
import paddle
from ppdiffusers import StableDiffusionPipeline
model_path = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_path)

learned_embeded_path = "./textual_inversion_cat/learned_embeds.pdparams"
for token, embeds in paddle.load(learned_embeded_path).items():
    pipe.tokenizer.add_tokens(token)
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
    token_id = pipe.tokenizer.convert_tokens_to_ids(token)
    with paddle.no_grad():
        pipe.text_encoder.get_input_embeddings().weight[token_id] = embeds

print(token)
# <cat-toy>
prompt = "A <cat-toy> backpack"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("cat-backpack.png")
