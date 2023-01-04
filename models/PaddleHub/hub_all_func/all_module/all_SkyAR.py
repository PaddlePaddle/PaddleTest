"""SkyAR"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_SkyAR_predict():
    """SkyAR predict"""
    os.system("hub install SkyAR")
    model = hub.Module(name="SkyAR")
    model.MagicSky(video_path="1.mp4", save_path="SkyAR_output")
    os.system("hub uninstall SkyAR")
