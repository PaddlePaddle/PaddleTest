"""word2vec_skipgram"""
import os
import paddle

import paddlehub as hub

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_word2vec_skipgram_predict():
    """word2vec_skipgram predict"""
    os.system("hub install word2vec_skipgram")
    # Load word2vec pretrained model
    module = hub.Module(name="word2vec_skipgram")
    inputs, outputs, program = module.context(trainable=True)

    # Must feed all the tensor of module need
    word_ids = inputs["text"]
    print("word_ids is: ", word_ids)

    # Use the pretrained word embeddings
    embedding = outputs["emb"]
    print("embedding is: ", embedding)
    os.system("hub uninstall word2vec_skipgram")
