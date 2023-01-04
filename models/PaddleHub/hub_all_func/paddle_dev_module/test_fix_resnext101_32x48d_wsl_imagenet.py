"""fix_resnext101_32x48d_wsl_imagenet"""
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


def test_fix_resnext101_32x48d_wsl_imagenet_predict():
    """fix_resnext101_32x48d_wsl_imagenet predict"""
    os.system("hub install fix_resnext101_32x48d_wsl_imagenet")
    classifier = hub.Module(name="fix_resnext101_32x48d_wsl_imagenet")
    images = [cv2.imread("doc_img.jpeg")]
    result = classifier.classification(images=images)
    print(result)
    os.system("hub uninstall fix_resnext101_32x48d_wsl_imagenet")
