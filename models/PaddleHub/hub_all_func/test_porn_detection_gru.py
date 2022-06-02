"""porn_detection_gru"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_porn_detection_gru_predict():
    """porn_detection_gru predict"""
    os.system("hub install porn_detection_gru")
    porn_detection_gru = hub.Module(name="porn_detection_gru")

    test_text = ["黄片下载", "打击黄牛党"]

    results = porn_detection_gru.detection(texts=test_text, use_gpu=use_gpu, batch_size=1)  # 如不使用GPU，请修改为use_gpu=False

    for index, text in enumerate(test_text):
        results[index]["text"] = text
    for index, result in enumerate(results):
        print(results[index])
    os.system("hub uninstall porn_detection_gru")
