"""resnet50_vd_dishes"""
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


def test_resnet50_vd_dishes_predict():
    """resnet50_vd_dishes predict"""
    os.system("hub install resnet50_vd_dishes")
    classifier = hub.Module(name="resnet50_vd_dishes")
    result = classifier.classification(images=[cv2.imread("doc_img.jpeg")])
    # or
    # result = classifier.classification(paths=['doc_img.jpeg'])
    print(result)
    os.system("hub uninstall resnet50_vd_dishes")
