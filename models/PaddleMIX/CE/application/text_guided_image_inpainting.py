from paddlemix.appflow import Appflow
from PIL import Image
from ppdiffusers.utils import load_image
img_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"
mask_url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations-mask.png"

image = load_image(img_url)
mask_image = load_image(mask_url)
paddle.seed(1024)

prompt = "Face of a yellow cat, high resolution, sitting on a park bench"

app = Appflow(app='inpainting',models=['stabilityai/stable-diffusion-2-inpainting'])
image = app(inpaint_prompt=prompt,image=image,seg_masks=mask_image)['result']

image.save("a_yellow_cat.png")