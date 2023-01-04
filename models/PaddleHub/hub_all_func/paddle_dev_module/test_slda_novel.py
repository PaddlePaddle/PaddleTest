"""slda_novel"""
import os
import paddle

import paddlehub as hub

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_slda_novel_predict():
    """slda_novel predict"""
    os.system("hub install slda_novel")
    slda_novel = hub.Module(name="slda_novel")

    topic_dist = slda_novel.infer_doc_topic_distribution("妈妈告诉女儿，今天爸爸过生日，放学后要早点回家一起庆祝")
    # [{'topic id': 222, 'distribution': 0.5}, {'topic id': 362, 'distribution': 0.5}]

    keywords = slda_novel.show_topic_keywords(topic_id=222)
    print("topic_dist is: ", topic_dist)
    print("keywords is: ", keywords)
    os.system("hub uninstall slda_novel")
