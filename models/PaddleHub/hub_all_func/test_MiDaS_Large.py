"""MiDaS_Large"""
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


def test_MiDaS_Large_predict():
    """MiDaS_Large predict"""
    os.system("hub install MiDaS_Large")
    model = hub.Module(name="MiDaS_Large")
    result = model.depth_estimation(images=[cv2.imread("doc_img.jpeg")])
    # or
    # result = model.depth_estimation(paths=['doc_img.jpeg'])
    print(result)
    os.system("hub uninstall MiDaS_Large")
