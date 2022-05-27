"""barometer_reader"""
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


def test_barometer_reader_predict():
    """
    barometer_reader
    :return:
    """
    os.system("hub install barometer_reader")
    model = hub.Module(name="barometer_reader")
    res = model.predict("doc_img.jpeg")
    print(res)
    os.system("hub uninstall barometer_reader")
