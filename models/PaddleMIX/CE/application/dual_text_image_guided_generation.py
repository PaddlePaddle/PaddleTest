from paddlemix.appflow import Appflow
from PIL import Image
from ppdiffusers.utils import load_image

url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/data/benz.jpg"
image = load_image(url)
prompt = "a red car in the sun"


app = Appflow(app='dual_text_and_image_guided_generation',models=['shi-labs/versatile-diffusion'])
image = app(prompt=prompt,image=image)['result']
image.save("versatile-diffusion-red_car.png")