"""spinalnet_res50_gemstone"""
import os
import paddle

import paddlehub as hub
import cv2

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_spinalnet_res50_gemstone_predict():
    """spinalnet_res50_gemstone predict"""
    os.system("hub install spinalnet_res50_gemstone")
    classifier = hub.Module(name="spinalnet_res50_gemstone")
    result = classifier.predict(["doc_img.jpeg"])
    print(result)
    os.system("hub uninstall spinalnet_res50_gemstone")
