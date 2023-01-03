"""Photo2Cartoon"""
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


def test_Photo2Cartoon_predict():
    """Photo2Cartoon predict"""
    os.system("hub install Photo2Cartoon")
    model = hub.Module(name="Photo2Cartoon")
    result = model.Cartoon_GEN(images=[cv2.imread("face_01.jpeg")])
    print(result)
    # or
    result = model.Cartoon_GEN(paths=["face_01.jpeg"])
    print(result)
    os.system("hub uninstall Photo2Cartoon")
