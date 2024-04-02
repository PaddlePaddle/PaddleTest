"""res2net101_vd_26w_4s_imagenet"""
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


def test_res2net101_vd_26w_4s_imagenet_predict():
    """res2net101_vd_26w_4s_imagenet predict"""
    os.system("hub install res2net101_vd_26w_4s_imagenet")
    classifier = hub.Module(name="res2net101_vd_26w_4s_imagenet")
    result = classifier.classification(images=[cv2.imread("doc_img.jpeg")])
    # or
    # result = classifier.classification(paths=['doc_img.jpeg'])
    print(result)
    os.system("hub uninstall res2net101_vd_26w_4s_imagenet")
