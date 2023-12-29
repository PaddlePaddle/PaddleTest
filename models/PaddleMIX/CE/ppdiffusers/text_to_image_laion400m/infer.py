from ppdiffusers import LDMTextToImagePipeline
model_name_or_path = "./ldm_pipelines"
pipe = LDMTextToImagePipeline.from_pretrained(model_name_or_path)
prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, guidance_scale=7.5).images[0]
image.save("astronaut_rides_horse.png")
