"""slda_news"""
import os
import paddle

import paddlehub as hub

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_slda_news_predict():
    """slda_news predict"""
    os.system("hub install slda_news")
    slda_news = hub.Module(name="slda_news")

    topic_dist = slda_news.infer_doc_topic_distribution("百度是全球最大的中文搜索引擎、致力于让网民更便捷地获取信息，找到所求。")
    # {378: 0.5, 804: 0.5}
    keywords = slda_news.show_topic_keywords(topic_id=804, k=10)
    print("topic_dist is: ", topic_dist)
    print("keywords is: ", keywords)
    os.system("hub uninstall slda_news")
