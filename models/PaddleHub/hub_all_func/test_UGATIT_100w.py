"""UGATIT_100w"""
import os
import paddlehub as hub
import paddle

import cv2

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_UGATIT_100w_predict():
    """UGATIT_100w predict"""
    os.system("hub install UGATIT_100w")
    model = hub.Module(name="UGATIT_100w")
    result = model.style_transfer(images=[cv2.imread("doc_img.jpeg")])
    # or
    # result = model.style_transfer(paths=['doc_img.jpeg'])
    print(result)
    os.system("hub uninstall UGATIT_100w")
