"""resnext101_vd_64x4d_imagenet"""
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


def test_resnext101_vd_64x4d_imagenet_predict():
    """resnext101_vd_64x4d_imagenet predict"""
    os.system("hub install resnext101_vd_64x4d_imagenet")
    classifier = hub.Module(name="resnext101_vd_64x4d_imagenet")
    test_img_path = "doc_img.jpeg"
    input_dict = {"image": [test_img_path]}
    result = classifier.classification(data=input_dict)
    print(result)
    os.system("hub uninstall resnext101_vd_64x4d_imagenet")
