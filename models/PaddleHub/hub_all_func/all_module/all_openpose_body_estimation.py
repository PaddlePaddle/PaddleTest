"""openpose_body_estimation"""
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


def test_openpose_body_estimation_predict():
    """openpose_body_estimation predict"""
    os.system("hub install openpose_body_estimation")
    model = hub.Module(name="openpose_body_estimation")
    result = model.predict("doc_img.jpeg")
    model.save_inference_model("openpose_body_estimation_model")
    print(result)
    os.system("hub uninstall openpose_body_estimation")
