"""deepspeech2_aishell"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_deepspeech2_aishell_predict():
    """deepspeech2_aishell"""
    os.system("hub install deepspeech2_aishell")
    # 采样率为16k，格式为wav的中文语音音频
    wav_file = "doc_audio_16000_zh.wav"
    model = hub.Module(name="deepspeech2_aishell")
    text = model.speech_recognize(wav_file)
    print(text)
    os.system("hub uninstall deepspeech2_aishell")
