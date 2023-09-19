from paddlemix.appflow import Appflow
from ppdiffusers.utils import load_image
task = Appflow(app="inpainting",
               models=["GroundingDino/groundingdino-swint-ogc","Sam/SamVitH-1024","stabilityai/stable-diffusion-2-inpainting"]
               )
paddle.seed(1024)
url = "https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg"
image_pil =  load_image(url)
result = task(image=image_pil,prompt="bus",inpaint_prompt="a yellow van")