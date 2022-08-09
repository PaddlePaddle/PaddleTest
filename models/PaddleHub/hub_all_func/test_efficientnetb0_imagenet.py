"""efficientnetb0_imagenet"""
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


def test_efficientnetb0_imagenet_predict():
    """efficientnetb0_imagenet"""
    os.system("hub install efficientnetb0_imagenet")
    classifier = hub.Module(name="efficientnetb0_imagenet")
    result = classifier.classification(images=[cv2.imread("doc_img.jpeg")])
    print(result)
    # or
    result = classifier.classification(paths=["doc_img.jpeg"])
    print(result)
    os.system("hub uninstall efficientnetb0_imagenet")
