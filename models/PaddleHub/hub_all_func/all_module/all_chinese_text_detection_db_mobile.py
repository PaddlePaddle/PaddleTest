"""chinese_text_detection_db_mobile"""
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


def test_chinese_text_detection_db_mobile_predict():
    """chinese_text_detection_db_mobile"""
    os.system("hub install chinese_text_detection_db_mobile")
    text_detector = hub.Module(name="chinese_text_detection_db_mobile", enable_mkldnn=True)  # mkldnn加速仅在CPU下有效
    result = text_detector.detect_text(images=[cv2.imread("doc_img.jpeg")])
    # or
    # result =text_detector.detect_text(paths=['doc_img.jpeg'])
    print(result)
    os.system("hub uninstall chinese_text_detection_db_mobile")
