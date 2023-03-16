"""hrnet48_imagenet"""
import os
import paddle

import paddlehub as hub

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_hrnet48_imagenet_predict():
    """hrnet48_imagenet predict"""
    os.system("hub install hrnet48_imagenet")
    model = hub.Module(name="hrnet48_imagenet")
    result = model.predict(["doc_img.jpeg"])
    print(result)
    os.system("hub uninstall hrnet48_imagenet")
