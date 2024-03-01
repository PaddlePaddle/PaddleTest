from ppdiffusers import DiffusionPipeline
from ppdiffusers import DDIMScheduler

import paddle

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", paddle_dtype=paddle.float16)
pipe.load_lora_weights("paddle_lora_weights.safetensors")

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
image = pipe("A picture of a sks dog in a bucket", num_inference_steps=25).images[0]
image.save("sks_dog.png")
