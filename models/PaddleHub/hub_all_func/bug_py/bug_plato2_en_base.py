"""plato2_en_base"""
import os
import paddle

import paddlehub as hub

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_plato2_en_base_predict():
    """plato2_en_base predict"""
    os.system("hub install plato2_en_base")
    module = hub.Module(name="plato2_en_base")

    test_texts = ["Hello", "Hello\thi, nice to meet you\tnice to meet you"]
    results = module.generate(texts=test_texts)
    for result in results:
        print(result)
    os.system("hub uninstall plato2_en_base")
