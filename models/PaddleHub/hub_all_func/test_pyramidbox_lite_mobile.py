"""pyramidbox_lite_mobile"""
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


def test_pyramidbox_lite_mobile_predict():
    """pyramidbox_lite_mobile predict"""
    os.system("hub install pyramidbox_lite_mobile")
    face_detector = hub.Module(name="pyramidbox_lite_mobile")
    result = face_detector.face_detection(images=[cv2.imread("doc_img.jpeg")])
    # or
    # result = face_detector.face_detection(paths=['doc_img.jpeg'])
    print(result)
    os.system("hub uninstall pyramidbox_lite_mobile")
