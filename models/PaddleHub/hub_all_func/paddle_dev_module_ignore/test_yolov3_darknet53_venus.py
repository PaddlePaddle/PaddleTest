"""yolov3_darknet53_venus"""
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


def test_yolov3_darknet53_venus_predict():
    """yolov3_darknet53_venus predict"""
    os.system("hub install yolov3_darknet53_venus")
    venus_detector = hub.Module(name="yolov3_darknet53_venus")
    result = venus_detector.object_detection(images=[cv2.imread("doc_img.jpeg")])
    # or
    # result = venus_detector.object_detection((paths=['doc_img.jpeg'])
    print(result)
    os.system("hub uninstall yolov3_darknet53_venus")
