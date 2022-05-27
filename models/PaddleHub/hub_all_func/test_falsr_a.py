"""falsr_a"""
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


def test_falsr_a_predict():
    """falsr_a"""
    os.system("hub install falsr_a")
    sr_model = hub.Module(name="falsr_a")
    im = cv2.imread("low_pixel.jpeg").astype("float32")
    # visualization=True可以用于查看超分图片效果，可设置为False提升运行速度。
    res = sr_model.reconstruct(images=[im], visualization=True)
    print(res[0]["data"])
    sr_model.save_inference_model()
    os.system("hub uninstall falsr_a")
