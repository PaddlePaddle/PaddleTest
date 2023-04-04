"""lda_news"""
import os
import paddle

import paddlehub as hub

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_lda_news_predict():
    """lda_news predict"""
    os.system("hub install lda_news")
    lda_news = hub.Module(name="lda_news")
    jsd, hd = lda_news.cal_doc_distance(doc_text1="今天的天气如何，适合出去游玩吗", doc_text2="感觉今天的天气不错，可以出去玩一玩了")
    print("jsd is: ", jsd)
    print("hd is: ", hd)

    lda_sim = lda_news.cal_query_doc_similarity(
        query="百度搜索引擎", document="百度是全球最大的中文搜索引擎、致力于让网民更便捷地获取信息，找到所求。百度超过千亿的中文网页数据库，可以瞬间找到相关的搜索结果。"
    )
    print("lda_sim is: ", lda_sim)

    results = lda_news.cal_doc_keywords_similarity("百度是全球最大的中文搜索引擎、致力于让网民更便捷地获取信息，找到所求。百度超过千亿的中文网页数据库，可以瞬间找到相关的搜索结果。")
    print(results)
    results = lda_news.infer_doc_topic_distribution("最近有学者新出了一篇论文，关于自然语言处理的，可厉害了")
    print(results)
    keywords = lda_news.show_topic_keywords(topic_id=216)
    print(keywords)
    os.system("hub uninstall lda_news")
