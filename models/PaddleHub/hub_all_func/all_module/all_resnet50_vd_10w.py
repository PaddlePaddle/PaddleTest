"""resnet50_vd_10w"""
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


def test_resnet50_vd_10w_predict():
    """resnet50_vd_10w predict"""
    os.system("hub install resnet50_vd_10w")
    classifier = hub.Module(name="resnet50_vd_10w")
    input_dict, output_dict, program = classifier.context(trainable=True)
    print(output_dict)
    os.system("hub uninstall resnet50_vd_10w")
