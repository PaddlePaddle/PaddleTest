

from ppdiffusers import StableDiffusionPipeline

model_path = "textual_inversion_cat"
pipe = StableDiffusionPipeline.from_pretrained(model_path)
prompt = "A <cat-toy> backpack"

image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("cat-backpack.png")