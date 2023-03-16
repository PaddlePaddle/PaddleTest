"""ernie_zeus"""
import os
import paddle

import paddlehub as hub

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_ernie_zeus_predict():
    """ernie_zeus"""
    os.system("hub install ernie_zeus")
    model = hub.Module(name="ernie_zeus")
    # 作文创作
    result = model.composition_generation(text="诚以养德，信以修身")
    print(result)
    os.system("hub uninstall ernie_zeus")
