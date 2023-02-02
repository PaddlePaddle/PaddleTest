"""faster_rcnn_resnet50_fpn_venus"""
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


def test_faster_rcnn_resnet50_fpn_venus_predict():
    """faster_rcnn_resnet50_fpn_venus predict"""
    os.system("hub install faster_rcnn_resnet50_fpn_venus")
    object_detector = hub.Module(name="faster_rcnn_resnet50_fpn_venus")
    result = object_detector.object_detection(images=[cv2.imread("doc_img.jpeg")])
    # or
    # result = object_detector.object_detection((paths=['doc_img.jpeg'])
    print(result)
    os.system("hub uninstall faster_rcnn_resnet50_fpn_venus")
