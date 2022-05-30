"""ocrnet_hrnetw18_cityscapes"""
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


def test_ocrnet_hrnetw18_cityscapes_predict():
    """ocrnet_hrnetw18_cityscapes predict"""
    os.system("hub install ocrnet_hrnetw18_cityscapes")
    model = hub.Module(name="ocrnet_hrnetw18_cityscapes")
    img = cv2.imread("doc_img.jpeg")
    model.predict(images=[img], visualization=True)
    os.system("hub uninstall ocrnet_hrnetw18_cityscapes")
