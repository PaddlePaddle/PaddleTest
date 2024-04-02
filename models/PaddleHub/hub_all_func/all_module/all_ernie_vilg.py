"""ernie_vilg"""
import os
import paddle

import paddlehub as hub

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_ernie_vilg_predict():
    """ernie_vilg"""
    os.system("hub install ernie_vilg")
    module = hub.Module(name="ernie_vilg")
    text_prompts = ["宁静的小镇"]
    images = module.generate_image(text_prompts=text_prompts, style="油画", output_dir="./ernie_vilg_out/")
    print(images)
    os.system("hub uninstall ernie_vilg")
