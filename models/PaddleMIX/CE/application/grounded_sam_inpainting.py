import paddle
from paddlemix.appflow import Appflow
from ppdiffusers.utils import load_image
task = Appflow(app="inpainting",
               models=["THUDM/chatglm-6b",
                       "GroundingDino/groundingdino-swint-ogc",
                       "Sam/SamVitH-1024",
                       "stabilityai/stable-diffusion-2-inpainting"]
               )
paddle.seed(1024)
url = "https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg"
image_pil = load_image(url)
inpaint_prompt = "bus is changed to A school bus parked on the roadside"
prompt = "Given caption,extract the main object to be replaced and marked it as 'main_object'," \
         + "Extract the remaining part as 'other prompt', " \
         + "Return main_object, other prompt in English" \
         + "Given caption: {}.".format(inpaint_prompt)
result = task(image=image_pil, prompt=prompt)
