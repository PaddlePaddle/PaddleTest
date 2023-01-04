"""fastspeech2_baker"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_fastspeech2_baker_predict():
    """fastspeech2_baker predict"""
    os.system("hub install fastspeech2_baker")
    sentences = ["这是一段测试语音合成的音频。"]
    model = hub.Module(name="fastspeech2_baker")
    wav_files = model.generate(sentences)
    # 打印合成的音频文件的路径
    print(wav_files)
    os.system("hub uninstall fastspeech2_baker")
