"""lda_novel"""
import os
import paddle
import paddlehub as hub

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_lda_novel_predict():
    """lda_novel predict"""
    os.system("hub install lda_novel")
    lda_novel = hub.Module(name="lda_novel")
    jsd, hd = lda_novel.cal_doc_distance(doc_text1="老人幸福地看着自己的儿子，露出了欣慰的笑容。", doc_text2="老奶奶看着自己的儿子，幸福地笑了。")
    print("jsd is: ", jsd)
    print("hd is: ", hd)

    lda_sim = lda_novel.cal_query_doc_similarity(query="亲孙女", document="老人激动地打量着面前的女孩，似乎找到了自己的亲孙女一般，双手止不住地颤抖着。")
    print("lda_sim is: ", lda_sim)

    results = lda_novel.cal_doc_keywords_similarity("百度是全球最大的中文搜索引擎、致力于让网民更便捷地获取信息，找到所求。百度超过千亿的中文网页数据库，可以瞬间找到相关的搜索结果。")
    print(results)
    results = lda_novel.infer_doc_topic_distribution("妈妈告诉女儿，今天爸爸过生日，放学后要早点回家一起庆祝")
    print(results)
    keywords = lda_novel.show_topic_keywords(topic_id=0)
    print(keywords)
    os.system("hub uninstall lda_novel")
