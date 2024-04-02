"""solov2"""
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


def test_solov2_predict():
    """solov2 predict"""
    os.system("hub install solov2")
    img = cv2.imread("doc_img.jpeg")
    model = hub.Module(name="solov2", use_gpu=use_gpu)
    output = model.predict(image=img, visualization=True)
    print(output)
    os.system("hub uninstall solov2")
