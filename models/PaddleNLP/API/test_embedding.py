"""Test_transformer is used to check Auto API test case."""
import io
import re
import warnings
import paddle
from paddlenlp.embeddings import TokenEmbedding
from paddlenlp.data import JiebaTokenizer


def getModelList():
    """
    get model links from
    <https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/model_zoo/embeddings.md>
    """
    model_list = []
    keyword = ["GB", "MB"]
    with open("./docs/model_zoo/embeddings.md") as f:
        for line in f:
            if any(word in line for word in keyword):
                result = re.split(r" | ", line)
                model_list.append(result[1])
        return model_list


def test_embedding():
    """
    download PretrainedModels & PretrainedTokenizer
    """
    model_list = getModelList()
    count = 0
    paddle.set_device("cpu")
    for model_name in model_list:
        count = count + 1
        try:
            print("processing:...{}/{}...{}".format(count, len(model_list), model_name))
            token_embedding = TokenEmbedding(embedding_name=model_name)
            search_embedding = token_embedding.search("中国")
            cosin_sim_embedding = token_embedding.cosine_sim("中国", "美国")
            dot_embedding = token_embedding.dot("中国", "美国")
            tokenizer = JiebaTokenizer(vocab=token_embedding.vocab)
            words = tokenizer.cut("中国人民")
            with io.open("./embedding_success.log", "a+", encoding="utf-8") as flog:
                flog.write(
                    "token_embedding:{},search_embedding:{},cosin_sim_embedding:{},dot_embedding:{},cut_embedding:{}".format(
                        token_embedding, search_embedding, cosin_sim_embedding, dot_embedding, words
                    )
                    + "\r\n"
                )
        except Exception as e:
            with io.open("./embedding_error.log", "a+", encoding="utf-8") as flog:
                flog.write("{}: {}".format(token_embedding, e) + "\r\n")


if __name__ == "__main__":
    test_embedding()
