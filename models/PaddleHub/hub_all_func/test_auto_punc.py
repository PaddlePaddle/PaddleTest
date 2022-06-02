"""auto_punc"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_auto_punc_predict():
    """
    auto_punc
    """
    os.system("hub install auto_punc")
    model = hub.Module(name="auto_punc")
    texts = ["今天的天气真好啊你下午有空吗我想约你一起去逛街", "我最喜欢的诗句是先天下之忧而忧后天下之乐而乐"]
    punc_texts = model.add_puncs(texts)
    print(punc_texts)
    os.system("hub uninstall auto_punc")
