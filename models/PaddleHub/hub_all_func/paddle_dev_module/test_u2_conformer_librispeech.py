"""u2_conformer_librispeech"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_u2_conformer_librispeech_predict():
    """u2_conformer_librispeech predict"""
    os.system("hub install u2_conformer_librispeech")
    # 采样率为16k，格式为wav的英文语音音频
    wav_file = "doc_audio_16000_en.wav"
    model = hub.Module(name="u2_conformer_librispeech")
    text = model.speech_recognize(wav_file)
    print(text)
    os.system("hub uninstall u2_conformer_librispeech")
