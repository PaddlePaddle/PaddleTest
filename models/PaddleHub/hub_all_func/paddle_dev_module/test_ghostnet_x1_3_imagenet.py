"""ghostnet_x1_0_imagenet"""
import os
import paddle

import paddlehub as hub

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_ghostnet_x1_0_imagenet_predict():
    """ghostnet_x1_0_imagenet predcit"""
    os.system("hub install ghostnet_x1_3_imagenet")
    model = hub.Module(name="ghostnet_x1_3_imagenet")
    result = model.predict(["doc_img.jpeg"])
    print(result)
    os.system("hub uninstall ghostnet_x1_3_imagenet")
