"""falsr_c"""
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


def test_falsr_c():
    """falsr_c"""
    os.system("hub install falsr_c")
    sr_model = hub.Module(name="falsr_c")
    im = cv2.imread("doc_img.jpeg").astype("float32")
    # visualization=True可以用于查看超分图片效果，可设置为False提升运行速度。
    res = sr_model.reconstruct(images=[im], visualization=True)
    print(res[0]["data"])
    os.system("hub uninstall falsr_c")
