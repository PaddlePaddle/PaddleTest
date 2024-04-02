"""hrnet40_imagenet"""
import os
import paddle

import paddlehub as hub

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_hrnet40_imagenet_predict():
    """hrnet40_imagenet predict"""
    os.system("hub install hrnet40_imagenet")
    model = hub.Module(name="hrnet40_imagenet")
    result = model.predict(["doc_img.jpeg"])
    print(result)
    os.system("hub uninstall hrnet40_imagenet")
