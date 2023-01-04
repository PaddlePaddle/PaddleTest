"""transformer_zh-en"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_transformer_zh_en_predict():
    """transformer_zh-en predict"""
    os.system("hub install transformer_zh-en")
    model = hub.Module(name="transformer_zh-en", beam_size=5)
    src_texts = ["今天天气怎么样？", "我们一起去吃饭吧。"]
    n_best = 3  # 每个输入样本的输出候选句子数量
    trg_texts = model.predict(src_texts, n_best=n_best)
    for idx, st in enumerate(src_texts):
        print("-" * 30)
        print(f"src: {st}")
        for i in range(n_best):
            print(f"trg[{i+1}]: {trg_texts[idx*n_best+i]}")
    os.system("hub uninstall transformer_zh-en")
