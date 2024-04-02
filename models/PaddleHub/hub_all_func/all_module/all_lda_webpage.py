"""lda_webpage"""
import os
import paddle

import paddlehub as hub

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_lda_webpage_predict():
    """lda_webpage predict"""
    os.system("hub install lda_webpage")
    lda_webpage = hub.Module(name="lda_webpage")
    jsd, hd = lda_webpage.cal_doc_distance(doc_text1="百度的网页上有着各种新闻的推荐，内容丰富多彩。", doc_text2="百度首页推荐着各种新闻，还提供了强大的搜索引擎功能。")
    print("jsd is: ", jsd)
    print("hd is: ", hd)

    results = lda_webpage.cal_doc_keywords_similarity("百度首页推荐着各种新闻，还提供了强大的搜索引擎功能。")
    print(results)
    out = lda_webpage.cal_query_doc_similarity(
        query="百度搜索引擎", document="百度是全球最大的中文搜索引擎、致力于让网民更便捷地获取信息，找到所求。百度超过千亿的中文网页数据库，可以瞬间找到相关的搜索结果。"
    )
    print(out)
    results = lda_webpage.infer_doc_topic_distribution("百度文库非常的好用，我们不仅在里面找到需要的文档，同时可以通过续费畅读精品文档。")
    print(results)
    keywords = lda_webpage.show_topic_keywords(3458)
    print(keywords)
    os.system("hub uninstall lda_webpage")
