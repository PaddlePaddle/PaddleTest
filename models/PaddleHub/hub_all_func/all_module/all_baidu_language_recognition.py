"""baidu_language_recognition"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_baidu_language_recognition_predict():
    """baidu_language_recognition predict"""
    os.system("hub install baidu_language_recognition")
    module = hub.Module(name="baidu_language_recognition")
    result = module.recognize("I like panda")
    print(result)
    os.system("hub uninstall baidu_language_recognition")
