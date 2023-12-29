from ppdiffusers import StableDiffusionPipeline
import paddle

model_path = "./sd-pokemon-model-lora"
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", paddle_dtype=paddle.float32)
# 注意：如果我们想从 HF Hub 加载权重，那么我们需要设置 from_hf_hub=True
pipe.unet.load_attn_procs(model_path)

prompt = "Totoro"
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("Totoro.png")
