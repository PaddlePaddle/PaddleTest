"""emotion_detection_textcnn"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_emotion_detection_textcnn_predict():
    """emotion_detection_textcnn"""
    os.system("hub install emotion_detection_textcnn")
    module = hub.Module(name="emotion_detection_textcnn")
    test_text = ["今天天气真好", "湿纸巾是干垃圾", "别来吵我"]
    results = module.emotion_classify(texts=test_text)
    for result in results:
        print(result["text"])
        print(result["emotion_label"])
        print(result["emotion_key"])
        print(result["positive_probs"])
        print(result["neutral_probs"])
        print(result["negative_probs"])
    os.system("hub uninstall emotion_detection_textcnn")
