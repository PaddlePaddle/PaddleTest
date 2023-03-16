"""chinese_ocr_db_crnn_server"""
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


def test_chinese_ocr_db_crnn_server_predict():
    """chinese_ocr_db_crnn_server"""
    os.system("hub install chinese_ocr_db_crnn_server")
    ocr = hub.Module(name="chinese_ocr_db_crnn_server", enable_mkldnn=True)  # mkldnn加速仅在CPU下有效
    result = ocr.recognize_text(images=[cv2.imread("doc_img.jpeg")])
    # or
    # result = ocr.recognize_text(paths=['doc_img.jpeg'])
    print(result)
    os.system("hub uninstall chinese_ocr_db_crnn_server")
