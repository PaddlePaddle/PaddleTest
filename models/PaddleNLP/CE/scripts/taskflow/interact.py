"""
taskflow api case
"""
import os

import numpy as np
import paddle
import paddlenlp
from paddlenlp import Taskflow


def test_knowledge_mining():
    """
    test knowledge mining
    """
    wordtag = Taskflow("knowledge_mining", model="wordtag", batch_size=2, max_seq_len=128, linking=True)
    wordtag("《孤女》是2010年九州出版社出版的小说，作者是余兼羽。")

    nptag = Taskflow("knowledge_mining", model="nptag", batch_size=2, max_seq_len=128, linking=True)
    nptag(["糖醋排骨", "红曲霉菌"])


def test_name_entity_recognition():
    """
    test name entity recognition
    """
    ner = Taskflow("ner", batch_size=2)
    ner("《长津湖》收尾，北美是最大海外票仓")
    ner_fast = Taskflow("ner", mode="fast")
    ner_fast("《长津湖》收尾，北美是最大海外票仓")
    ner_entity = Taskflow("ner", mode="accurate", entity_only=True)
    ner_entity("《孤女》是2010年九州出版社出版的小说，作者是余兼羽")


def test_word_segmetation():
    """
    test word segmetation
    """
    seg = Taskflow("word_segmentation", batch_size=2)
    seg(["第十四届全运会在西安举办", "三亚是一个美丽的城市"])
    seg_fast = Taskflow("word_segmentation", mode="fast")
    seg_fast(["第十四届全运会在西安举办", "三亚是一个美丽的城市"])
    seg_acc = Taskflow("word_segmentation", mode="accurate")
    seg_acc("李伟拿出具有科学性、可操作性的《陕西省高校管理体制改革实施方案》")


def test_pos_tagging():
    """
    test pos tagging
    """
    tag = Taskflow("pos_tagging", batch_size=2)
    tag("第十四届全运会在西安举办")


def test_corrector():
    """
    test corrector
    """
    corrector = Taskflow("text_correction", batch_size=2)
    corrector("遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。")


def test_dependency_parsing():
    """
    test dependency parsing
    """
    ddp = Taskflow("dependency_parsing", model="ddparser")
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
    test sentiment analysis
    """
    skep = Taskflow
    print("skep", skep)


if __name__ == "__main__":
    exit_code = 0
    try:
        test_knowledge_mining()
    except:
        exit_code = 1
        print("test_knowledge_mining Failed")
    try:
        test_name_entity_recognition()
    except:
        exit_code = 1
        print("test_name_entity_recognition Failed")
    try:
        test_pos_tagging()
    except:
        exit_code = 1
        print("test_pos_tagging Failed")
    try:
        test_corrector()
    except:
        exit_code = 1
        print("test_corrector Failed")
    try:
        test_word_segmetation()
    except:
        exit_code = 1
        print("test_word_segmetation Failed")
    try:
        test_dependency_parsing()
    except:
        exit_code = 1
        print("test_dependency_parsing Failed")
    try:
        test_sentiment_analysis()
    except:
        exit_code = 1
        print("test_sentiment_analysis Failed")
    print("exit_code: %s" % exit_code)
