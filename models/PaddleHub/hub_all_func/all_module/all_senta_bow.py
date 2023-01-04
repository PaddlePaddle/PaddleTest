"""senta_bow"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_senta_bow_predict():
    """senta_bow predict"""
    os.system("hub install senta_bow")
    senta = hub.Module(name="senta_bow")
    test_text = ["这家餐厅很好吃", "这部电影真的很差劲"]
    results = senta.sentiment_classify(texts=test_text, use_gpu=use_gpu, batch_size=1)
    for result in results:
        print(result["text"])
        print(result["sentiment_label"])
        print(result["sentiment_key"])
        print(result["positive_probs"])
        print(result["negative_probs"])

    # 这家餐厅很好吃 1 positive 0.9782 0.0218
    # 这部电影真的很差劲 0 negative 0.0124 0.9876
    os.system("hub uninstall senta_bow")
