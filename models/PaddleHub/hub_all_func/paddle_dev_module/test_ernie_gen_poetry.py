"""ernie_gen_poetry"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_ernie_gen_poetry():
    """ernie_gen_poetry"""
    os.system("hub install ernie_gen_poetry")
    module = hub.Module(name="ernie_gen_poetry")

    test_texts = ["昔年旅南服，始识王荆州。", "高名出汉阴，禅阁跨香岑。"]
    results = module.generate(texts=test_texts, use_gpu=use_gpu, beam_width=5)
    for result in results:
        print(result)
    os.system("hub uninstall ernie_gen_poetry")
