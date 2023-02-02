"""transformer_tts_ljspeech"""
import os
import paddlehub as hub
import soundfile as sf
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_transformer_tts_ljspeech_predict():
    """transformer_tts_ljspeech predict"""
    os.system("hub install transformer_tts_ljspeech")
    # Load transformer_tts_ljspeech module.
    module = hub.Module(name="transformer_tts_ljspeech")

    # Predict sentiment label
    test_texts = ["Life was like a box of chocolates, you never know what you're gonna get."]
    wavs, sample_rate = module.synthesize(texts=test_texts, use_gpu=use_gpu, vocoder="waveflow")
    for index, wav in enumerate(wavs):
        sf.write(f"{index}.wav", wav, sample_rate)
    os.system("hub uninstall transformer_tts_ljspeech")
