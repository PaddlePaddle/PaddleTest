"""wav2lip"""
import os
import paddle

import paddlehub as hub

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_wav2lip_predict():
    """wav2lip predict"""
    os.system("hub install wav2lip")
    module = hub.Module(name="wav2lip")
    face_input_path = "doc_img.jpeg"
    audio_input_path = "doc_audio.wav"
    module.wav2lip_transfer(
        face=face_input_path, audio=audio_input_path, output_dir="wav2lip_transfer_result", use_gpu=use_gpu
    )
    os.system("hub uninstall wav2lip")
