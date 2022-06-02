"""ernie_gen_acrostic_poetry"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_ernie_gen_acrostic_poetry_predict():
    """ernie_gen_acrostic_poetry"""
    os.system("hub install ernie_gen_acrostic_poetry")
    # 在模型定义时，可以通过设置line=4或8指定输出绝句或律诗，设置word=5或7指定输出五言或七言。
    # 默认line=4, word=7 即输出七言绝句。
    module = hub.Module(name="ernie_gen_acrostic_poetry", line=4, word=7)

    test_texts = ["我喜欢你"]
    results = module.generate(texts=test_texts, use_gpu=use_gpu, beam_width=5)
    for result in results:
        print(result)
    os.system("hub uninstall ernie_gen_acrostic_poetry")
