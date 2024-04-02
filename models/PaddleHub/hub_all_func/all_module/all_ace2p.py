"""ace2p"""
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


def test_ace2p_predict():
    """ace2p predict"""
    os.system("hub install ace2p")
    human_parser = hub.Module(name="ace2p")
    result = human_parser.segmentation(images=[cv2.imread("doc_img.jpeg")])
    print(result)
    os.system("hub uninstall ace2p")
