"""pp-tinypose"""
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


def test_pp_tinypose_predict():
    """pp-tinypose predict"""
    os.system("hub install pp-tinypose")
    model = hub.Module(name="pp-tinypose")
    result = model.predict("doc_img.jpeg", save_path="pp_tinypose_output", visualization=True, use_gpu=use_gpu)
    print(result)
    os.system("hub uninstall pp-tinypose")
