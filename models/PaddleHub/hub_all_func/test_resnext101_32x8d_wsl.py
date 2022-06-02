"""resnext101_32x8d_wsl"""
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


def test_resnext101_32x8d_wsl_predict():
    """resnext101_32x8d_wsl predict"""
    os.system("hub install resnext101_32x8d_wsl")
    classifier = hub.Module(name="resnext101_32x8d_wsl")
    test_img_path = "doc_img.jpeg"
    input_dict = {"image": [test_img_path]}
    result = classifier.classification(data=input_dict)
    print(result)
    os.system("hub uninstall resnext101_32x8d_wsl")
