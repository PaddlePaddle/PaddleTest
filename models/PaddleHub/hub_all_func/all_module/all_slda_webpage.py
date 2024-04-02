"""slda_webpage"""
import os
import paddle

import paddlehub as hub

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_slda_webpage_predict():
    """slda_webpage predict"""
    os.system("hub install slda_webpage")
    slda_webpage = hub.Module(name="slda_webpage")

    topic_dist = slda_webpage.infer_doc_topic_distribution("百度是全球最大的中文搜索引擎、致力于让网民更便捷地获取信息，找到所求。")

    keywords = slda_webpage.show_topic_keywords(topic_id=4687)
    print("topic_dist is: ", topic_dist)
    print("keywords is: ", keywords)
    os.system("hub uninstall slda_webpage")
