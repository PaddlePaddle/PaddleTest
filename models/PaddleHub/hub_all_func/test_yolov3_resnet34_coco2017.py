"""yolov3_resnet34_coco2017"""
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


def test_yolov3_resnet34_coco2017_predict():
    """yolov3_resnet34_coco2017 predict"""
    os.system("hub install yolov3_resnet34_coco2017")
    object_detector = hub.Module(name="yolov3_resnet34_coco2017")
    result = object_detector.object_detection(images=[cv2.imread("doc_img.jpeg")])
    # or
    # result = object_detector.object_detection((paths=['doc_img.jpeg'])
    print(result)
    os.system("hub uninstall yolov3_resnet34_coco2017")
