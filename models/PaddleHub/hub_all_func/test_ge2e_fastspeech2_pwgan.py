"""ge2e_fastspeech2_pwgan"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_ge2e_fastspeech2_pwgan_predict():
    """ge2e_fastspeech2_pwgan predict"""
    os.system("hub install ge2e_fastspeech2_pwgan")
    model = hub.Module(
        name="ge2e_fastspeech2_pwgan", output_dir="ge2e_fastspeech2_pwgan_result", speaker_audio="doc_audio.wav"
    )  # 指定目标音色音频文件
    texts = ["语音的表现形式在未来将变得越来越重要$", "今天的天气怎么样$"]
    wavs = model.generate(texts, use_gpu=use_gpu)
    for text, wav in zip(texts, wavs):
        print("=" * 30)
        print(f"Text: {text}")
        print(f"Wav: {wav}")
    os.system("hub uninstall ge2e_fastspeech2_pwgan")
