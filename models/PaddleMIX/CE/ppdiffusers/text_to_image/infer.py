from ppdiffusers import StableDiffusionPipeline
model_path = "sd-pokemon-model"
pipe = StableDiffusionPipeline.from_pretrained(model_path)

image = pipe(prompt="yoda").images[0]
image.save("yoda-pokemon.png")
