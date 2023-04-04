"""MiDaS_Small"""
import os
import paddle

import cv2
import paddlehub as hub

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_MiDaS_Small_predict():
    """MiDaS_Small predict"""
    os.system("hub install MiDaS_Small")
    # 模型加载
    # use_gpu：是否使用GPU进行预测
    model = hub.Module(name="MiDaS_Small", use_gpu=use_gpu)
    # 模型预测
    result = model.depth_estimation(images=[cv2.imread("doc_img.jpeg")])
    # or
    # result = model.style_transfer(paths=['doc_img.jpeg'])
    print(result)
    os.system("hub uninstall MiDaS_Small")
