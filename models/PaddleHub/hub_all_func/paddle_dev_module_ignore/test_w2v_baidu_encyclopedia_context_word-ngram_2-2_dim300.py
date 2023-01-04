"""w2v_baidu_encyclopedia_context_word-ngram_2-2_dim300"""
import os
import paddle

import paddlehub as hub

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_w2v_baidu_encyclopedia_context_word_ngram_2_2_dim300_predict():
    """w2v_baidu_encyclopedia_context_word-ngram_2-2_dim300"""
    os.system("hub install w2v_baidu_encyclopedia_context_word-ngram_2-2_dim300")
    embedding = hub.Module(name="w2v_baidu_encyclopedia_context_word-ngram_2-2_dim300")

    # 获取单词的embedding
    embedding.search("中国")
    # 计算两个词向量的余弦相似度
    embedding.cosine_sim("中国", "美国")
    # 计算两个词向量的内积
    embedding.dot("中国", "美国")
    os.system("hub uninstall w2v_baidu_encyclopedia_context_word-ngram_2-2_dim300")
