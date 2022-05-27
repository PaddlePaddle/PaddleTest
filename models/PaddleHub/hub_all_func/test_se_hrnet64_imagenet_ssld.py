"""se_hrnet64_imagenet_ssld"""
import os
import paddle

import paddlehub as hub

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_se_hrnet64_imagenet_ssld_predict():
    """se_hrnet64_imagenet_ssld predict"""
    os.system("hub install se_hrnet64_imagenet_ssld")
    model = hub.Module(name="se_hrnet64_imagenet_ssld")
    result = model.predict(["doc_img.jpeg"])
    print(result)
    os.system("hub uninstall se_hrnet64_imagenet_ssld")
