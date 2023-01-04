"""lstm_tacotron2"""
import os
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_lstm_tacotron2_predict():
    """lstm_tacotron2 predict"""
    os.system("hub install lstm_tacotron2")
    model = hub.Module(
        name="lstm_tacotron2",
        output_dir="lstm_tacotron2_data",
        speaker_audio=os.path.join("lstm_tacotron2_data", "man.wav"),
    )  # 指定目标音色音频文件
    texts = ["语音的表现形式在未来将变得越来越重要$", "今天的天气怎么样$"]
    wavs = model.generate(texts, use_gpu=True)
    for text, wav in zip(texts, wavs):
        print("=" * 30)
        print(f"Text: {text}")
        print(f"Wav: {wav}")
    os.system("hub uninstall lstm_tacotron2")
