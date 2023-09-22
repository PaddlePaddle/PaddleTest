import paddle
from IPython.display import display
from ppdiffusers import AutoencoderKL, StableDiffusionImg2ImgPipeline
from ppdiffusers.utils import load_image
from ppdiffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import preprocess

def decode_image(image):
    image = (image / 2 + 0.5).clip(0, 1).transpose([0, 2, 3, 1]).cast('float32').numpy()
    image = StableDiffusionImg2ImgPipeline.numpy_to_pil(image)
    return image

model_name_or_path = "./autoencoder_outputs/checkpoint-100"
vae = AutoencoderKL.from_pretrained(model_name_or_path)
image = load_image("https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/vermeer.jpg")
sample_32 = preprocess(image.resize((256, 256)))
sample_64 = preprocess(image.resize((512, 512)))

with paddle.no_grad():
    # sample_32 256 x 256
    dec_32 = vae(sample_32, sample_posterior=True)[0] # must set sample_posterior = True
    img_32 = decode_image(dec_32)[0]
    display(img_32)
    # img_32 512 x 512
    img_32.save('32.jpg')

with paddle.no_grad():
    # sample_32 512 x 512
    dec_64 = vae(sample_64, sample_posterior=True)[0] # must set sample_posterior = True
    img_64 = decode_image(dec_64)[0]
    display(img_64)
    # img_64 1024 x 1024
    img_64.save('64.jpg')