"""baidu_translate"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_baidu_translate_predict():
    """baidu_translate predict"""
    os.system("hub install baidu_translate")
    module = hub.Module(name="baidu_translate")
    result = module.translate("I like panda")
    print(result)
    os.system("hub uninstall baidu_translate")
