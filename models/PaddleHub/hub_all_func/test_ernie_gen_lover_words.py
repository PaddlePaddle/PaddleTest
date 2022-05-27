"""ernie_gen_lover_words"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_ernie_gen_lover_words():
    """ernie_gen_lover_words"""
    os.system("hub install ernie_gen_lover_words")
    module = hub.Module(name="ernie_gen_lover_words")

    test_texts = ["情人节", "故乡", "小编带大家了解一下程序员情人节"]
    results = module.generate(texts=test_texts, use_gpu=use_gpu, beam_width=5)
    for result in results:
        print(result)
    os.system("hub uninstall ernie_gen_lover_words")
