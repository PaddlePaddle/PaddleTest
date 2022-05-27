"""efficientnetb2_imagenet"""
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


def test_efficientnetb2_imagenet_predict():
    """efficientnetb2_imagenet"""
    os.system("hub install efficientnetb2_imagenet")
    classifier = hub.Module(name="efficientnetb2_imagenet")
    result = classifier.classification(images=[cv2.imread("doc_img.jpeg")])
    # or
    # result = classifier.classification(paths=['doc_img.jpeg'])
    print(result)
    os.system("hub install efficientnetb2_imagenet")
