"""CPM_LM"""
import os
import paddle

import paddlehub as hub

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_CPM_LM_predict():
    """CPM_LM"""
    os.system("python -m pip install --force-reinstall sentencepiece==0.1.92")
    os.system("hub install CPM_LM")
    model = hub.Module(name="CPM_LM")
    # 作文创作
    inputs = """默写古诗:
    日照香炉生紫烟，遥看瀑布挂前川。
    飞流直下三千尺，"""
    outputs = model.predict(inputs, max_len=10, end_word="\n")
    print(inputs + outputs)
    os.system("hub uninstall CPM_LM")
    os.system("python -m pip uninstall sentencepiece -y")
    os.system("python -m pip install sentencepiece")
