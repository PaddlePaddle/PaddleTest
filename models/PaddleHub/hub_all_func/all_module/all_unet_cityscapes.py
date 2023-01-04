"""unet_cityscapes"""
import os
import paddle

import cv2
import paddlehub as hub

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_unet_cityscapes_predict():
    """unet_cityscapes predict"""
    os.system("hub install unet_cityscapes")
    model = hub.Module(name="unet_cityscapes")
    img = cv2.imread("doc_img.jpeg")
    model.predict(images=[img], visualization=True)
    os.system("hub uninstall unet_cityscapes")
