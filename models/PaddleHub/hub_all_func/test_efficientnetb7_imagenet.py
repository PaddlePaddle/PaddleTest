"""efficientnetb7_imagenet"""
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


def test_efficientnetb7_imagenet_predict():
    """efficientnetb7_imagenet"""
    os.system("hub install efficientnetb7_imagenet")
    classifier = hub.Module(name="efficientnetb7_imagenet")
    result = classifier.classification(images=[cv2.imread("doc_img.jpeg")])
    # or
    # result = classifier.classification(paths=['doc_img.jpeg'])
    print(result)
    os.system("hub uninstall efficientnetb7_imagenet")
