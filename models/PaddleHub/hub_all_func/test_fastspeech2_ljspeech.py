"""fastspeech2_ljspeech"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_fastspeech2_ljspeech_predict():
    """fastspeech2_ljspeech"""
    # 需要合成语音的文本
    os.system("hub install fastspeech2_ljspeech")
    sentences = ["The quick brown fox jumps over a lazy dog."]
    model = hub.Module(name="fastspeech2_ljspeech")
    wav_files = model.generate(sentences)
    # 打印合成的音频文件的路径
    print(wav_files)
    os.system("hub uninstall fastspeech2_ljspeech")
