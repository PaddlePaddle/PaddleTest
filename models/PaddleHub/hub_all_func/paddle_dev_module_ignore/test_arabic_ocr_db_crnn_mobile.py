"""arabic_ocr_db_crnn_mobile"""
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


def test_arabic_ocr_db_crnn_mobile_predict():
    """
    arabic_ocr_db_crnn_mobile
    """
    os.system("hub install arabic_ocr_db_crnn_mobile")
    ocr = hub.Module(name="arabic_ocr_db_crnn_mobile", enable_mkldnn=True)  # mkldnn加速仅在CPU下有效
    result = ocr.recognize_text(images=[cv2.imread("doc_img.jpeg")])
    # or
    # result = ocr.recognize_text(paths=['doc_img.jpeg'])
    print(result)
    os.system("hub uninstall arabic_ocr_db_crnn_mobile")
