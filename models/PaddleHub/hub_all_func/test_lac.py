"""lac"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_lac_predict():
    """lac predict"""
    os.system("hub install lac")
    lac = hub.Module(name="lac")
    test_text = ["今天是个好日子", "天气预报说今天要下雨", "下一班地铁马上就要到了"]
    results = lac.cut(text=test_text, use_gpu=use_gpu, batch_size=1, return_tag=True)
    for result in results:
        print(result["word"])
        print(result["tag"])
    os.system("hub uninstall lac")
