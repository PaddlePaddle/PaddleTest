"""chinese_text_detection_db_server"""
import os
import paddle

import paddlehub as hub
import cv2

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_chinese_text_detection_db_server_predict():
    """chinese_text_detection_db_server"""
    os.system("hub install chinese_text_detection_db_server")
    text_detector = hub.Module(name="chinese_text_detection_db_server")
    result = text_detector.detect_text(images=[cv2.imread("doc_img.jpeg")])
    print(result)
    os.system("hub uninstall chinese_text_detection_db_server")
