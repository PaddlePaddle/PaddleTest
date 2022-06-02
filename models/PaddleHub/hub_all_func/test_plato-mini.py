"""plato-mini"""
import os

# 非交互模式
import paddle

import paddlehub as hub

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_plato_mini_predict():
    """plato-mini predict"""
    os.system("hub install plato-mini")
    model = hub.Module(name="plato-mini")
    data = [["你是谁？"], ["你好啊。", "吃饭了吗？"]]
    result = model.predict(data)
    print(result)
    os.system("hub uninstall plato-mini")
