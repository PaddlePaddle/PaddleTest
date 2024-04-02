"""UGATIT_92w"""
import os
import cv2
import paddle

import paddlehub as hub

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_UGATIT_92w_predict():
    """UGATIT_92w predict"""
    os.system("hub install UGATIT_92w")
    # 模型加载
    # use_gpu：是否使用GPU进行预测
    model = hub.Module(name="UGATIT_92w", use_gpu=use_gpu)
    # 模型预测
    result = model.style_transfer(images=[cv2.imread("doc_img.jpeg")])
    # or
    # result = model.style_transfer(paths=['doc_img.jpeg'])
    print(result)
    os.system("hub uninstall UGATIT_92w")
