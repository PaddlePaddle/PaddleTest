"""repvgg_b1g2_imagenet"""
import os
import paddle

import paddlehub as hub

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_repvgg_b1g2_imagenet_predict():
    """repvgg_b1g2_imagenet predict"""
    os.system("hub install repvgg_b1g2_imagenet")
    model = hub.Module(name="repvgg_b1g2_imagenet")
    result = model.predict(["doc_img.jpeg"])
    print(result)
    os.system("hub uninstall repvgg_b1g2_imagenet")
