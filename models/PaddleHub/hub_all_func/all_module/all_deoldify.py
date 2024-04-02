"""deoldify"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_deoldify():
    """deoldify"""
    os.system("hub install deoldify")
    model = hub.Module(name="deoldify")
    model.predict("black_white.jpeg")
    # model.predict('1.mp4')
    os.system("hub uninstall deoldify")
