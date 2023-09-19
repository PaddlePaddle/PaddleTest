from paddlemix import Appflow
from ppdiffusers.utils import load_image
task = Appflow(app="auto_label",
               models=["paddlemix/blip2-caption-opt2.7b","GroundingDino/groundingdino-swint-ogc","Sam/SamVitH-1024"]
               )
url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"
image_pil = load_image(url)
blip2_prompt = 'describe the image'
result = task(image=image_pil,blip2_prompt = blip2_prompt)