"""multi_languages_ocr_db_crnn"""
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


def test_multi_languages_ocr_db_crnn_predict():
    """multi_languages_ocr_db_crnn predict"""
    os.system("hub install multi_languages_ocr_db_crnn")
    ocr = hub.Module(name="multi_languages_ocr_db_crnn", lang="en", enable_mkldnn=True)  # mkldnn加速仅在CPU下有效
    result = ocr.recognize_text(images=[cv2.imread("doc_img.jpeg")])
    # or
    # result = ocr.recognize_text(paths=['doc_img.jpeg'])
    print(result)
    os.system("hub uninstall multi_languages_ocr_db_crnn")
