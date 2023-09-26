from ppdiffusers import StableDiffusionAdapterPipeline, T2IAdapter
from ppdiffusers.utils import load_image
adapter = T2IAdapter.from_pretrained("./sd15_openpose/checkpoint-12000/adapter")
pipe = StableDiffusionAdapterPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", adapter = adapter, safety_checker=None)
pose_image = load_image("https://paddlenlp.bj.bcebos.com/models/community/westfish/t2i-adapter/test/man-openpose.png")
img = pipe(prompt="a beautiful girl", image=pose_image, guidance_scale=9, num_inference_steps=50).images[0]
img.save("demo.png")