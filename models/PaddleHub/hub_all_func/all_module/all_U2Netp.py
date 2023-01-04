"""U2Netp"""
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


def test_U2Netp_predict():
    """U2Netp predict"""
    os.system("hub install U2Netp")
    model = hub.Module(name="U2Netp")
    result = model.Segmentation(
        images=[cv2.imread("doc_img.jpeg")],
        paths=None,
        batch_size=1,
        input_size=320,
        output_dir="U2Netp_output",
        visualization=True,
    )
    print(result)
    os.system("hub uninstall U2Netp")
