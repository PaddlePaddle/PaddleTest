"""Vehicle_License_Plate_Recognition"""
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


def test_Vehicle_License_Plate_Recognition_predict():
    """Vehicle_License_Plate_Recognition predict"""
    os.system("hub install Vehicle_License_Plate_Recognition")
    model = hub.Module(name="Vehicle_License_Plate_Recognition")
    result = model.plate_recognition(images=[cv2.imread("doc_img.jpeg")])
    print(result)
    os.system("hub uninstall Vehicle_License_Plate_Recognition")
