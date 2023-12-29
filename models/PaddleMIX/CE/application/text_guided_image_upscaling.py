from paddlemix.appflow import Appflow
from PIL import Image
from ppdiffusers.utils import load_image

url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/low_res_cat.png"

low_res_img = load_image(url).resize((128, 128))

prompt = "a white cat"

app = Appflow(app='image2image_text_guided_upscaling', models=[
              'stabilityai/stable-diffusion-x4-upscaler'])
image = app(prompt=prompt, image=low_res_img)['result']

image.save("upscaled_white_cat.png")
