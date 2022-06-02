"""ernie_skep_sentiment_analysis"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_ernie_skep_sentiment_analysis_predict():
    """ernie_skep_sentiment_analysis"""
    os.system("hub install ernie_skep_sentiment_analysis")
    # Load ernie_skep_sentiment_analysis module.
    module = hub.Module(name="ernie_skep_sentiment_analysis")
    # Predict sentiment label
    test_texts = ["你不是不聪明，而是不认真", "虽然小明很努力，但是他还是没有考100分"]
    results = module.predict_sentiment(test_texts, use_gpu=use_gpu)

    for result in results:
        print(result["text"])
        print(result["sentiment_label"])
        print(result["positive_probs"])
        print(result["negative_probs"])
    os.system("hub uninstall ernie_skep_sentiment_analysis")
