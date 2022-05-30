"""fcn_hrnetw48_cityscapes"""
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


def test_fcn_hrnetw48_cityscapes_predict():
    """fcn_hrnetw48_cityscapes predict"""
    os.system("hub install fcn_hrnetw48_cityscapes")
    model = hub.Module(name="fcn_hrnetw48_cityscapes")
    img = cv2.imread("doc_img.jpeg")
    model.predict(images=[img], visualization=True)
    os.system("hub uninstall fcn_hrnetw48_cityscapes")
