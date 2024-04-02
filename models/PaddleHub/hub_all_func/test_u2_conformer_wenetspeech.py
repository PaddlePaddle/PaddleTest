"""u2_conformer_wenetspeech"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_u2_conformer_wenetspeech_predict():
    """u2_conformer_wenetspeech predict"""
    os.system("hub install u2_conformer_wenetspeech")
    # 采样率为16k，格式为wav的中文语音音频
    wav_file = "doc_audio.wav"
    model = hub.Module(name="u2_conformer_wenetspeech")
    text = model.speech_recognize(wav_file)
    print(text)
    os.system("hub uninstall u2_conformer_wenetspeech")
