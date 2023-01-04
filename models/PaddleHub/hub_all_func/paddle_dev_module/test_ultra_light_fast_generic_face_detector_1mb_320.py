"""ultra_light_fast_generic_face_detector_1mb_320"""
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


def test_ultra_light_fast_generic_face_detector_1mb_320_predict():
    """ultra_light_fast_generic_face_detector_1mb_320 predict"""
    os.system("hub install ultra_light_fast_generic_face_detector_1mb_320")
    face_detector = hub.Module(name="ultra_light_fast_generic_face_detector_1mb_320")
    result = face_detector.face_detection(images=[cv2.imread("doc_img.jpeg")])
    # or
    # result = face_detector.face_detection(paths=['doc_img.jpeg'])
    print(result)
    os.system("hub uninstall ultra_light_fast_generic_face_detector_1mb_320")
