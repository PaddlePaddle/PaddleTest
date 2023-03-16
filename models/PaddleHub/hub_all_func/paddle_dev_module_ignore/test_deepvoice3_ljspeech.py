"""deepvoice3_ljspeech"""
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


def test_deepvoice3_ljspeech_predict():
    """deepvoice3_ljspeech predict"""
    os.system("hub install deepvoice3_ljspeech")
    # Load deepvoice3_ljspeech module.
    module = hub.Module(name="deepvoice3_ljspeech")

    # Predict sentiment label
    test_texts = [
        "Simple as this proposition is, it is necessary to be stated",
        "Parakeet stands for Paddle PARAllel text-to-speech toolkit",
    ]
    wavs, sample_rate = module.synthesize(texts=test_texts)
    for index, wav in enumerate(wavs):
        sf.write(f"{index}.wav", wav, sample_rate)
    os.system("hub uninstall deepvoice3_ljspeech")
