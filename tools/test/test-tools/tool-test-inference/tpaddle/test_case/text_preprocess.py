# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
text preprocess
"""
import paddlenlp as ppnlp


def ernie_data(data_path):
    """
    ernie data text to list(array).
    Args:
       data_path(str): data path
    Returns:
       examples(list): array in list
    """
    max_seq_length = 128
    examples = []
    with open(data_path, "r", encoding="utf-8") as f:
        data = f.readlines()
        tokenizer = ppnlp.transformers.ErnieTinyTokenizer.from_pretrained("ernie-tiny")
        for text in data:
            encoded_inputs = tokenizer(text=text, max_seq_len=max_seq_length)
            input_ids = encoded_inputs["input_ids"]
            token_type_ids = encoded_inputs["token_type_ids"]
            examples.append((input_ids, token_type_ids))
        f.close
    return examples
