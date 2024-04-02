"""rexnet_1_0_imagenet"""
import os
import paddle

import paddlehub as hub

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_rexnet_1_0_imagenet_predict():
    """rexnet_1_0_imagenet predict"""
    os.system("hub install rexnet_1_0_imagenet")
    model = hub.Module(name="rexnet_1_0_imagenet")
    result = model.predict(["doc_img.jpeg"])
    print(result)
    os.system("hub uninstall rexnet_1_0_imagenet")
