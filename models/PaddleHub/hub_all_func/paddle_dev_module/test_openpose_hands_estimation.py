"""openpose_hands_estimation"""
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


def test_openpose_hands_estimation_predict():
    """openpose_hands_estimation predict"""
    os.system("hub install openpose_hands_estimation")
    model = hub.Module(name="openpose_hands_estimation")
    result = model.predict("doc_img.jpeg")
    model.save_inference_model("openpose_hands_estimation_model")
    print(result)
    os.system("hub uninstall openpose_hands_estimation")
