"""deeplabv3p_xception65_humanseg"""
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


def test_deeplabv3p_xception65_humanseg_predict():
    """deeplabv3p_xception65_humanseg"""
    os.system("hub install deeplabv3p_xception65_humanseg")
    human_seg = hub.Module(name="deeplabv3p_xception65_humanseg")
    result = human_seg.segmentation(images=[cv2.imread("doc_img.jpeg")])
    print(result)
    os.system("hub uninstall deeplabv3p_xception65_humanseg")
