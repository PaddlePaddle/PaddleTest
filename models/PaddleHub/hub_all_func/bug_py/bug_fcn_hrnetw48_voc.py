"""fcn_hrnetw48_voc"""
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


def test_fcn_hrnetw48_voc_predict():
    """fcn_hrnetw48_voc predict"""
    os.system("hub install fcn_hrnetw48_voc")
    model = hub.Module(name="fcn_hrnetw48_voc")
    img = cv2.imread("doc_img.jpeg")
    model.predict(images=[img], visualization=True)
    os.system("hub uninstall fcn_hrnetw48_voc")
