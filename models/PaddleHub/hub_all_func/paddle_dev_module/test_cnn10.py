"""panns_cnn10"""
# ESC50声音分类预测
import os
import paddle

import librosa
import paddlehub as hub
from paddlehub.datasets import ESC50

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_panns_cnn10_predict():
    """panns_cnn10"""
    os.system("hub install panns_cnn10")
    sr = 44100  # 音频文件的采样率
    wav_file = "doc_audio.wav"  # 用于预测的音频文件路径
    checkpoint = "model.pdparams"  # 用于预测的模型参数
    label_map = {idx: label for idx, label in enumerate(ESC50.label_list)}
    model = hub.Module(
        name="panns_cnn10", task="sound-cls", num_class=ESC50.num_class, label_map=label_map, load_checkpoint=checkpoint
    )
    data = [librosa.load(wav_file, sr=sr)[0]]
    result = model.predict(data, sample_rate=sr, batch_size=1, feat_type="mel", use_gpu=use_gpu)
    print("File: {}\tLable: {}".format(wav_file, result[0]))
    os.system("hub uninstall panns_cnn10")
