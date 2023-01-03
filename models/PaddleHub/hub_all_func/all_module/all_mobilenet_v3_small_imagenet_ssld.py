"""mobilenet_v3_small_imagenet_ssld"""
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


def test_mobilenet_v3_small_imagenet_ssld_predict():
    """mobilenet_v3_small_imagenet_ssld predict"""
    os.system("hub install mobilenet_v3_small_imagenet_ssld")
    classifier = hub.Module(name="mobilenet_v3_small_imagenet_ssld")
    result = classifier.classification(images=[cv2.imread("doc_img.jpeg")])
    # or
    # result = classifier.classification(paths=['doc_img.jpeg'])
    print(result)
    os.system("hub uninstall mobilenet_v3_small_imagenet_ssld")
