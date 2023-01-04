"""hrnet64_imagenet"""
import os
import paddle

import paddlehub as hub

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_hrnet64_imagenet_predict():
    """hrnet64_imagenet"""
    os.system("hub install hrnet64_imagenet")
    model = hub.Module(name="hrnet64_imagenet")
    result = model.predict(["doc_img.jpeg"])
    print(result)
    os.system("hub uninstall hrnet64_imagenet")
