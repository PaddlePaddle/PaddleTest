"""msgnet"""
import os
import paddle

import paddlehub as hub

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_msgnet_predict():
    """msgnet"""
    os.system("hub install msgnet")
    model = hub.Module(name="msgnet")
    result = model.predict(
        origin=["doc_img.jpeg"], style="doc_img.jpeg", visualization=True, save_path="msgnet_style_tranfer"
    )
    print(result)
    os.system("hub uninstall msgnet")
