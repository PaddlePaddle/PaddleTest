"""face_landmark_localization"""
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


def test_face_landmark_localization_predict():
    """face_landmark_localization"""
    os.system("hub install face_landmark_localization")
    face_landmark = hub.Module(name="face_landmark_localization")

    # Replace face detection module to speed up predictions but reduce performance
    # face_landmark.set_face_detector_module(hub.Module(name="ultra_light_fast_generic_face_detector_1mb_320"))

    result = face_landmark.keypoint_detection(images=[cv2.imread("doc_img.jpeg")])
    # or
    # result = face_landmark.keypoint_detection(paths=['doc_img.jpeg'])
    print(result)
    os.system("hub uninstall face_landmark_localization")
