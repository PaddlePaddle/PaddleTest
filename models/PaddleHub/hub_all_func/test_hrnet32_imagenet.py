"""hrnet32_imagenet"""
import os
import paddle

import paddlehub as hub

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_hrnet32_imagenet_predict():
    """hrnet32_imagenet predict"""
    os.system("hub install hrnet32_imagenet")
    model = hub.Module(name="hrnet32_imagenet")
    result = model.predict(["doc_img.jpeg"])
    print(result)
    os.system("hub uninstall hrnet32_imagenet")
