"""senta_bilstm"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_senta_bilstm_predict():
    """senta_bilstm predict"""
    os.system("hub install senta_bilstm")
    senta = hub.Module(name="senta_bilstm")
    test_text = ["这家餐厅很好吃", "这部电影真的很差劲"]
    results = senta.sentiment_classify(texts=test_text, use_gpu=use_gpu, batch_size=1)

    for result in results:
        print(result["text"])
        print(result["sentiment_label"])
        print(result["sentiment_key"])
        print(result["positive_probs"])
        print(result["negative_probs"])

    # 这家餐厅很好吃 1 positive 0.9407 0.0593
    # 这部电影真的很差劲 0 negative 0.02 0.98
    os.system("hub uninstall senta_bilstm")
