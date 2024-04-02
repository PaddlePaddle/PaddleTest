"""Extract_Line_Draft"""
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


def test_Extract_Line_Draft_predict():
    """Extract_Line_Draft predict"""
    os.system("hub install Extract_Line_Draft")
    Extract_Line_Draft_test = hub.Module(name="Extract_Line_Draft")
    test_img = "face_01.jpeg"
    # execute predict
    results = Extract_Line_Draft_test.ExtractLine(test_img, use_gpu=use_gpu)
    print(results)
    os.system("hub uninstall Extract_Line_Draft")
