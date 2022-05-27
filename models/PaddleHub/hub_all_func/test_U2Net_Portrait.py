"""U2Net_Portrait"""
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


def test_U2Net_Portrait_predict():
    """U2Net_Portrait predict"""
    os.system("hub install U2Net_Portrait")
    model = hub.Module(name="U2Net_Portrait")
    result = model.Portrait_GEN(images=[cv2.imread("doc_img.jpeg")])
    # or
    # result = model.Portrait_GEN(paths=['doc_img.jpeg'])
    print(result)
    os.system("hub uninstall U2Net_Portrait")
