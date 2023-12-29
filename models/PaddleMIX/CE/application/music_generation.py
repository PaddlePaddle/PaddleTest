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
music_generation
"""
# music generation
from paddlemix.appflow import Appflow
import paddle
from PIL import Image
import scipy
paddle.seed(1024)

# Text to music
task = Appflow(app="music_generation", models=["cvssp/audioldm"])
prompt = "A classic cocktail lounge vibe with smooth jazz piano and a cool, relaxed atmosphere."
negative_prompt = 'low quality, average quality, muffled quality, noise interference, poor and low-grade quality, inaudible quality, low-fidelity quality'
audio_length_in_s = 5
num_inference_steps = 20
output_path = "tmp.wav"
result = task(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps,
              audio_length_in_s=audio_length_in_s, generator=paddle.Generator().manual_seed(120))['result']
scipy.io.wavfile.write(output_path, rate=16000, data=result)

# image to music
prompt = "Given the scene description in the following paragraph, please create a musical style sentence that fits the scene.  Description:{}.".format(
    result)
task2 = Appflow(app="music_generation", models=[
                "THUDM/chatglm-6b", "cvssp/audioldm"])
result = task2(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps,
               audio_length_in_s=audio_length_in_s, generator=paddle.Generator().manual_seed(120))['result']
scipy.io.wavfile.write(output_path, rate=16000, data=result)
