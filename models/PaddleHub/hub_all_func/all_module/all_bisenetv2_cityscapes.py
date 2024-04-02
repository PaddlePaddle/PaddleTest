"""bisenetv2_cityscapes"""
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


def test_bisenetv2_cityscapes_predict():
    """bisenetv2_cityscapes"""
    os.system("hub install bisenetv2_cityscapes")
    model = hub.Module(name="bisenetv2_cityscapes")
    img = cv2.imread("doc_img.jpeg")
    model.predict(images=[img], visualization=True)
    os.system("hub uninstall bisenetv2_cityscapes")
