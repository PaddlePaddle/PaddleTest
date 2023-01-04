"""dpn131_imagenet"""
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


def test_dpn131_imagenet_predict():
    """dpn131_imagenet"""
    os.system("hub install dpn131_imagenet")
    classifier = hub.Module(name="dpn131_imagenet")
    test_img_path = "doc_img.jpeg"
    input_dict = {"image": [test_img_path]}
    result = classifier.classification(data=input_dict)
    print(result)
    os.system("hub uninstall dpn131_imagenet")
