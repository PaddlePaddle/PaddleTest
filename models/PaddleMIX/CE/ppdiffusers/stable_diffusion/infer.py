from ppdiffusers import StableDiffusionPipeline, UNet2DConditionModel
# 加载上面我们训练好的unet权重
unet_model_name_or_path = "./laion400m_pretrain_output_trainer/unet"
unet = UNet2DConditionModel.from_pretrained(unet_model_name_or_path)
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None, unet=unet)
prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, guidance_scale=7.5, width=256, height=256).images[0]
image.save("astronaut_rides_horse.png")