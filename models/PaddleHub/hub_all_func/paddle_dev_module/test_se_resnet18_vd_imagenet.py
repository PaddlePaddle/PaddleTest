"""se_resnet18_vd_imagenet"""
import os
import paddle

import paddlehub as hub
import cv2

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_se_resnet18_vd_imagenet_predict():
    """se_resnet18_vd_imagenet predict"""
    os.system("hub install se_resnet18_vd_imagenet")
    classifier = hub.Module(name="se_resnet18_vd_imagenet")

    result = classifier.classification(images=[cv2.imread("doc_img.jpeg")])
    # or
    # result = classifier.classification(paths=['doc_img.jpeg'])
    print(result)
    os.system("hub uninstall se_resnet18_vd_imagenet")
