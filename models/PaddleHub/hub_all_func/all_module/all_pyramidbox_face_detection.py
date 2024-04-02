"""pyramidbox_face_detection"""
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


def test_pyramidbox_face_detection_predict():
    """pyramidbox_face_detection predict"""
    os.system("hub install pyramidbox_face_detection")
    face_detector = hub.Module(name="pyramidbox_face_detection")
    result = face_detector.face_detection(images=[cv2.imread("doc_img.jpeg")])
    # or
    # result = face_detector.face_detection(paths=['doc_img.jpeg'])
    print(result)
    os.system("hub uninstall pyramidbox_face_detection")
