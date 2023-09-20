from paddlemix.appflow import Appflow
from ppdiffusers.utils import load_image
import matplotlib.pyplot as plt
import numpy as np

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

task = Appflow(app="openset_det_sam",
                   models=["GroundingDino/groundingdino-swint-ogc","Sam/SamVitH-1024"],
                   static_mode=False) #如果开启静态图推理，设置为True,默认动态图
url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"
image_pil = load_image(url)
result = task(image=image_pil,prompt="dog")

plt.figure(figsize=(10, 10))
plt.imshow(image_pil)
for mask in result['seg_masks']:
    show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)


plt.axis('off')
plt.savefig(
    'dog.jpg',
    bbox_inches="tight", dpi=300, pad_inches=0.0
)