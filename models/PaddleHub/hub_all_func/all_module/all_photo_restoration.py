"""photo_restoration"""
import os
import cv2
import paddle

import paddlehub as hub

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_photo_restoration_predict():
    """photo_restoration predict"""
    os.system("hub install photo_restoration")
    model = hub.Module(name="photo_restoration", visualization=True)
    im = cv2.imread("black_white.jpeg")
    res = model.run_image(im)
    print(res)
    os.system("hub uninstall photo_restoration")
