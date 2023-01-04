"""ch_pp-ocrv3"""
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


def test_ch_pp_ocrv3_det_predict():
    """
    ch_pp-ocrv3
    """
    os.system("hub install ch_pp-ocrv3")
    ocr = hub.Module(name="ch_pp-ocrv3", enable_mkldnn=True)  # mkldnn加速仅在CPU下有效
    result = ocr.recognize_text(images=[cv2.imread("doc_img.jpeg")])
    print(result)
    os.system("hub uninstall ch_pp-ocrv3")
