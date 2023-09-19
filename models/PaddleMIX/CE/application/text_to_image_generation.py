import paddle
from paddlemix.appflow import Appflow

paddle.seed(1024)
task = Appflow(app="text2image_generation",
               models=["stabilityai/stable-diffusion-v1-5"]
               )
prompt = "a photo of an astronaut riding a horse on mars."
result = task(prompt=prompt)['result']