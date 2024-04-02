"""user_guided_colorization"""
import os
import paddle

import paddlehub as hub

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_user_guided_colorization_predict():
    """user_guided_colorization predict"""
    os.system("hub install user_guided_colorization")
    model = hub.Module(name="user_guided_colorization")
    model.set_config(prob=0.1)
    result = model.predict(images=["doc_img.jpeg"])
    print(result)
    os.system("hub uninstall user_guided_colorization")
