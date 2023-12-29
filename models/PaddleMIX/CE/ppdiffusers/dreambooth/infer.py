from ppdiffusers import StableDiffusionPipeline
model_path = "./dream_outputs"
pipe = StableDiffusionPipeline.from_pretrained(model_path)

prompt = "A photo of sks dog in a bucket"
image = pipe(prompt).images[0]
image.save("sks-dog.png")
