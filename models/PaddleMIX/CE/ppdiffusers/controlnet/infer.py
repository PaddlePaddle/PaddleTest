from ppdiffusers import StableDiffusionControlNetPipeline, ControlNetModel
from ppdiffusers.utils import load_image
controlnet = ControlNetModel.from_pretrained("./sd15_control/checkpoint-12000/controlnet")
pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet = controlnet, safety_checker=None)
canny_edged_image = load_image("221844474-fd539851-7649-470e-bded-4d174271cc7f.png")
img = pipe(prompt="pale golden rod circle with old lace background", image=canny_edged_image, guidance_scale=9, num_inference_steps=50).images[0]
img.save("demo.png")