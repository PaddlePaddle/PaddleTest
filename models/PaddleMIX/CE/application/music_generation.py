#music generation
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
result = task(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps, audio_length_in_s=audio_length_in_s, generator = paddle.Generator().manual_seed(120))['result']
scipy.io.wavfile.write(output_path, rate=16000, data=result)

# image to music
task1 = Appflow(app="music_generation", models=["miniGPT4/MiniGPT4-7B"])
negative_prompt = 'low quality, average quality, muffled quality, noise interference, poor and low-grade quality, inaudible quality, low-fidelity quality'  
audio_length_in_s = 5
num_inference_steps = 20
output_path = "tmp.wav"
minigpt4_text = 'describe the image, '
image_pil = Image.open("dance.png").convert("RGB")
result = task1(image=image_pil, minigpt4_text=minigpt4_text )['result'].split('#')[0]
paddle.device.cuda.empty_cache()

prompt = "Given the scene description in the following paragraph, please create a musical style sentence that fits the scene.  Description:{}.".format(result)
task2 = Appflow(app="music_generation", models=["THUDM/chatglm-6b", "cvssp/audioldm"])
result = task2(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps, audio_length_in_s=audio_length_in_s, generator = paddle.Generator().manual_seed(120))['result']
scipy.io.wavfile.write(output_path, rate=16000, data=result)