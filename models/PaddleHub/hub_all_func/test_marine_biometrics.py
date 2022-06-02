"""marine_biometrics"""
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


def test_marine_biometrics_predict():
    """marine_biometrics predict"""
    os.system("hub install marine_biometrics")
    classifier = hub.Module(name="marine_biometrics")
    images = [cv2.imread("doc_img.jpeg")]
    results = classifier.predict(images=images)
    for result in results:
        print(result)
    os.system("hub uninstall marine_biometrics")
