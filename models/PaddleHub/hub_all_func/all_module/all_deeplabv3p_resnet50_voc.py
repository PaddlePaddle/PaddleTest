"""deeplabv3p_resnet50_voc"""
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


def test_deeplabv3p_resnet50_voc_predict():
    """deeplabv3p_resnet50_voc"""
    os.system("hub install deeplabv3p_resnet50_voc")
    model = hub.Module(name="deeplabv3p_resnet50_voc")
    img = cv2.imread("doc_img.jpeg")
    model.predict(images=[img], visualization=True)
    os.system("hub uninstall deeplabv3p_resnet50_voc")
