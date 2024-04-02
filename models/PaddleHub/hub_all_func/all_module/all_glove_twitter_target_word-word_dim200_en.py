"""glove_twitter_target_word-word_dim200_en"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_glove_twitter_target_word_word_dim200_en():
    """glove_twitter_target_word-word_dim200_en predict"""
    os.system("hub install glove_twitter_target_word-word_dim200_en")
    embedding = hub.Module(name="glove_twitter_target_word-word_dim200_en")
    # 获取单词的embedding
    embedding.search("中国")
    # 计算两个词向量的余弦相似度
    embedding.cosine_sim("中国", "美国")
    # 计算两个词向量的内积
    embedding.dot("中国", "美国")
    os.system("hub uninstall glove_twitter_target_word-word_dim200_en")
