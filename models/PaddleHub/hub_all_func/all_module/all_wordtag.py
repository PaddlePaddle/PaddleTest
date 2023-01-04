"""wordtag"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_wordtag_predict():
    """wordtag predict"""
    os.system("hub install wordtag")
    # Load WordTag
    module = hub.Module(name="wordtag")
    # String input
    results = module.predict("《孤女》是2010年九州出版社出版的小说，作者是余兼羽。")
    print(results)
    # List input
    results = module.predict(["热梅茶是一道以梅子为主要原料制作的茶饮", "《孤女》是2010年九州出版社出版的小说，作者是余兼羽"])
    print(results)
    os.system("hub uninstall wordtag")
