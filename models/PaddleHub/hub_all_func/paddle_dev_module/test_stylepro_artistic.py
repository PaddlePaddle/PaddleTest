"""stylepro_artistic"""
import os
import paddle

import paddlehub as hub
import cv2

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_stylepro_artistic_predict():
    """stylepro_artistic predict"""
    os.system("hub install stylepro_artistic")
    stylepro_artistic = hub.Module(name="stylepro_artistic")
    result = stylepro_artistic.style_transfer(
        images=[{"content": cv2.imread("doc_img.jpeg"), "styles": [cv2.imread("doc_img.jpeg")]}]
    )
    print(result)
    os.system("hub uninstall stylepro_artistic")
