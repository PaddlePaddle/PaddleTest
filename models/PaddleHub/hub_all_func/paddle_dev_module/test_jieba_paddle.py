"""jieba_paddle"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_jieba_paddle_predict():
    """jieba_paddle predict"""
    os.system("hub install jieba_paddle")
    jieba = hub.Module(name="jieba_paddle")

    results = jieba.cut("今天是个好日子", cut_all=False, HMM=True)
    print(results)
    os.system("hub uninstall jieba_paddle")

    # ['今天', '是', '个', '好日子']
