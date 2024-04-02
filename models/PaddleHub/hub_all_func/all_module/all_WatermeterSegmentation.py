"""WatermeterSegmentation"""
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


def test_WatermeterSegmentation_predict():
    """WatermeterSegmentation predict"""
    os.system("hub install WatermeterSegmentation")
    seg = hub.Module(name="WatermeterSegmentation")
    res = seg.cutPic(picUrl="doc_img.jpeg")
    print(res)
    os.system("hub uninstall WatermeterSegmentation")
