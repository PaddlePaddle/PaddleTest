"""slda_weibo"""
import os
import paddle

import paddlehub as hub

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_slda_weibo_predict():
    """slda_weibo predict"""
    os.system("hub install slda_weibo")
    slda_weibo = hub.Module(name="slda_weibo")

    topic_dist = slda_weibo.infer_doc_topic_distribution("百度是全球最大的中文搜索引擎、致力于让网民更便捷地获取信息，找到所求。")
    # [{'topic id': 874, 'distribution': 0.5}, {'topic id': 1764, 'distribution': 0.5}]

    keywords = slda_weibo.show_topic_keywords(topic_id=874)

    print("topic_dist is: ", topic_dist)
    print("keywords is: ", keywords)
    os.system("hub uninstall slda_weibo")
