"""yolov3_darknet53_pedestrian"""
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


def test_yolov3_darknet53_pedestrian_predict():
    """yolov3_darknet53_pedestrian predict"""
    os.system("hub install yolov3_darknet53_pedestrian")
    pedestrian_detector = hub.Module(name="yolov3_darknet53_pedestrian")
    result = pedestrian_detector.object_detection(images=[cv2.imread("doc_img.jpeg")])
    # or
    # result = pedestrian_detector.object_detection(paths=['doc_img.jpeg'])
    print(result)
    os.system("hub uninstall yolov3_darknet53_pedestrian")
