"""ernie_tiny_couplet"""
import os
import paddle

import paddlehub as hub

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_ernie_tiny_couplet_predict():
    """ernie_tiny_couplet"""
    # Load ernie pretrained model
    os.system("hub install ernie_tiny_couplet")
    module = hub.Module(name="ernie_tiny_couplet", use_gpu=use_gpu)
    results = module.generate(["风吹云乱天垂泪", "若有经心风过耳"])
    for result in results:
        print(result)
    os.system("hub uninstall ernie_tiny_couplet")
