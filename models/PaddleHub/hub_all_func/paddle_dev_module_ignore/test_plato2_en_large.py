"""plato2_en_large"""
import os
import paddle

import paddlehub as hub

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_plato2_en_large_predict():
    """plato2_en_large predict"""
    os.system("hub install plato2_en_large")
    module = hub.Module(name="plato2_en_large")

    test_texts = ["Hello", "Hello\thi, nice to meet you\tnice to meet you"]
    results = module.generate(texts=test_texts)
    for result in results:
        print(result)
    os.system("hub uninstall plato2_en_large")


def test_plato2_en_large_interactive_mode():
    """plato2_en_large predict"""
    os.system("hub install plato2_en_large")
    module = hub.Module(name="plato2_en_large")

    with module.interactive_mode(max_turn=6):
        while True:
            human_utterance = input("[Human]: ").strip()
            robot_utterance = module.generate(human_utterance)
            print("[Bot]: %s" % robot_utterance[0])
    os.system("hub uninstall plato2_en_large")
