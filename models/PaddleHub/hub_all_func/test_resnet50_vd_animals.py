"""resnet50_vd_animals"""
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


def test_resnet50_vd_animals_predict():
    """resnet50_vd_animals predict"""
    os.system("hub install resnet50_vd_animals")
    classifier = hub.Module(name="resnet50_vd_animals")
    result = classifier.classification(images=[cv2.imread("doc_img.jpeg")])
    # or
    # result = classifier.classification(paths=['doc_img.jpeg'])
    print(result)
    os.system("hub uninstall resnet50_vd_animals")
