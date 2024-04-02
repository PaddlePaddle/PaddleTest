"""stgan_bald"""
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


def test_stgan_bald_predict():
    """stgan_bald predict"""
    os.system("hub install stgan_bald")
    stgan_bald = hub.Module(name="stgan_bald")
    result = stgan_bald.bald(images=[cv2.imread("face_01.jpeg")])
    print(result)
    os.system("hub uninstall stgan_bald")
