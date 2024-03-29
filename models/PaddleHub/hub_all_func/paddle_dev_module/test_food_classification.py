"""food_classification"""
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


def test_food_classification_predict():
    """food_classification predict"""
    os.system("hub install food_classification")
    classifier = hub.Module(name="food_classification")
    images = [cv2.imread("doc_img.jpeg")]
    results = classifier.predict(images=images)
    for result in results:
        print(result)
    os.system("hub uninstall food_classification")
