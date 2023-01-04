"""hrnet48_imagenet_ssld"""
import os
import paddle

import paddlehub as hub

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_hrnet48_imagenet_ssld_predict():
    """hrnet48_imagenet_ssld predict"""
    os.system("hub install hrnet48_imagenet_ssld")
    model = hub.Module(name="hrnet48_imagenet_ssld")
    result = model.predict(["doc_img.jpeg"])
    print(result)
    os.system("hub uninstall hrnet48_imagenet_ssld")
