import paddle
from ppdiffusers import StableDiffusionPipeline
model_path = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_path)

learned_embeded_path = "./textual_inversion_cat/learned_embeds.pdparams"
for token, embeds in paddle.load(learned_embeded_path).items():
    pipe.tokenizer.add_tokens(token)
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
    token_id = pipe.tokenizer.convert_tokens_to_ids(token)
    with paddle.no_grad():
        pipe.text_encoder.get_input_embeddings().weight[token_id] = embeds

print(token)
# <cat-toy>
prompt = "A <cat-toy> backpack"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("cat-backpack.png")