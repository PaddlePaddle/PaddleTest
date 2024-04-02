"""videotag_tsn_lstm"""
import os
import cv2
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_videotag_tsn_lstm_predict():
    """videotag_tsn_lstm predict"""
    os.system("hub install videotag_tsn_lstm")
    videotag = hub.Module(name="videotag_tsn_lstm")

    # execute predict and print the result
    results = videotag.classify(paths=["1.mp4"], use_gpu=use_gpu)  # 示例文件请在上方下载
    print(results)
    os.system("hub uninstall videotag_tsn_lstm")
