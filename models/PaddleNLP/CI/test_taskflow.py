"""Test taskflow."""
import os
import paddle
import paddlenlp

import numpy as np
from paddlenlp import Taskflow


def test_knowledge_mining():
    """
    taskflow knowledge_mining test case
    """
    wordtag = Taskflow("knowledge_mining", model="wordtag", batch_size=2, max_seq_len=128, linking=True)
    wordtag("《孤女》是2010年九州出版社出版的小说，作者是余兼羽。")
    nptag = Taskflow("knowledge_mining", model="nptag", batch_size=2, max_seq_len=128, linking=True)
    nptag(["糖醋排骨", "红曲霉菌"])


def test_name_entity_recognition():
    """
    taskflow name_entity_recognition test case
    """
    ner = Taskflow("ner", batch_size=2)
    ner("《长津湖》收尾，北美是最大海外票仓")
    ner_fast = Taskflow("ner", mode="fast")
    ner_fast("《长津湖》收尾，北美是最大海外票仓")
    ner_entity = Taskflow("ner", mode="accurate", entity_only=True)
    ner_entity("《孤女》是2010年九州出版社出版的小说，作者是余兼羽")


def test_word_segmetation():
    """
    taskflow word_segmetation test case
    """
    seg = Taskflow("word_segmentation", batch_size=2)
    seg(["第十四届全运会在西安举办", "三亚是一个美丽的城市"])
    seg_fast = Taskflow("word_segmentation", mode="fast")
    seg_fast(["第十四届全运会在西安举办", "三亚是一个美丽的城市"])
    seg_acc = Taskflow("word_segmentation", mode="accurate")
    seg_acc("李伟拿出具有科学性、可操作性的《陕西省高校管理体制改革实施方案》")


def test_pos_tagging():
    """
    taskflow pos_tagging test case
    """
    tag = Taskflow("pos_tagging", batch_size=2)
    tag("第十四届全运会在西安举办")


def test_corrector():
    """
    taskflow corrector test case
    """
    corrector = Taskflow("text_correction", batch_size=2)
    corrector("遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。")


def test_dependency_parsing():
    """
    taskflow dependency_parsing test case
    """
    ddp = Taskflow("dependency_parsing", model="ddparser", batch_size=2, prob=True, use_pos=True)
    print(ddp("9月9日上午纳达尔在亚瑟·阿什球场击败俄罗斯球员梅德韦杰夫"))
    print(ddp.from_segments([["9月9日", "上午", "纳达尔", "在", "亚瑟·阿什球场", "击败", "俄罗斯", "球员", "梅德韦杰夫"]]))
    ddp_ernie = Taskflow("dependency_parsing", model="ddparser-ernie-1.0", batch_size=2, prob=True, use_pos=True)
    print(ddp_ernie("9月9日上午纳达尔在亚瑟·阿什球场击败俄罗斯球员梅德韦杰夫"))
    ddp_ernie_gram = Taskflow(
        "dependency_parsing", model="ddparser-ernie-gram-zh", batch_size=2, prob=True, use_pos=True
    )
    print(ddp_ernie_gram("9月9日上午纳达尔在亚瑟·阿什球场击败俄罗斯球员梅德韦杰夫"))


def test_sentiment_analysis():
    """
    taskflow sentiment_analysis test case
    """
    skep = Taskflow("sentiment_analysis", batch_size=2)
    skep("这个产品用起来真的很流畅，我非常喜欢")

    skep_ernie = Taskflow("sentiment_analysis", model="skep_ernie_1.0_large_ch", batch_size=2)
    skep_ernie("作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。")


def test_text_similarity():
    """
    taskflow text_similarity test case
    """
    similarity = Taskflow("text_similarity", batch_size=2)
    similarity([["世界上什么东西最小", "世界上什么东西最小？"]])


def test_question_answering():
    """
    taskflow question_answering test case
    """
    qa = Taskflow("question_answering", batch_size=2)
    qa("中国的国土面积有多大？")


def test_poetry():
    """
    taskflow poetry test case
    """
    poetry = Taskflow("poetry_generation", batch_size=2)
    poetry("林密不见人")


def test_dialogue():
    """
    taskflow dialogue test case
    """
    dialogue = Taskflow("dialogue", batch_size=2, max_seq_len=512)
    dialogue(["吃饭了吗"])


def test_uie():
    """
    taskflow uie test case
    """
    schema_ner = ["时间", "选手", "赛事名称"]  # Define the schema for entity extraction
    ie = Taskflow("information_extraction", schema=schema_ner, model="uie-large", batch_size=2, prob=True, use_pos=True)
    ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！")

    ie = Taskflow("information_extraction", schema=schema_ner, model="uie-tiny", batch_size=2, prob=True, use_pos=True)
    schema_re = {"歌曲名称": ["歌手", "所属专辑"]}  # Define the schema for relation extraction
    ie.set_schema(schema_re)  # Reset schema
    ie("《告别了》是孙耀威在专辑爱的故事里面的歌曲")

    ie = Taskflow("information_extraction", schema=schema_ner, prob=True, use_pos=True)
    schema_ee = {"歌曲名称": ["歌手", "所属专辑"]}  # Define the schema for relation extraction
    ie.set_schema(schema_ee)  # Reset schema
    ie("《告别了》是孙耀威在专辑爱的故事里面的歌曲")

    schema_opinion = {"评价维度": "观点词"}  # Define the schema for opinion extraction
    ie.set_schema(schema_opinion)  # Reset schema
    ie("个人觉得管理太混乱了，票价太高了")

    schema_sa = "情感倾向[正向，负向]"  # Define the schema for sentence-level sentiment classification
    ie.set_schema(schema_sa)  # Reset schema
    ie("这个产品用起来真的很流畅，我非常喜欢")
    # [{'情感倾向[正向，负向]': [{'text': '正向', 'probability': 0.9990110458312529}]}]

    schema_bre = ["寺庙", {"丈夫": "妻子"}]
    ie.set_schema(schema_bre)
    ie("李治即位后，让身在感业寺的武则天续起头发，重新纳入后宫。")


if __name__ == "__main__":
    test_knowledge_mining()
    test_name_entity_recognition()
    test_word_segmetation()
    test_pos_tagging()
    test_corrector()
    test_dependency_parsing()
    test_sentiment_analysis()
    test_text_similarity()
    test_question_answering()
    test_poetry()
    test_dialogue()
    test_uie()
