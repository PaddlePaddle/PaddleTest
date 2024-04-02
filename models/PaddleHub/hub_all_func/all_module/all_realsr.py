"""realsr"""
import os
import cv2
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_realsr_predict():
    """realsr predict"""
    os.system("hub install realsr")
    model = hub.Module(name="realsr")
    results = model.predict("low_pixel.jpeg")
    print(results)
    os.system("hub uninstall realsr")
