"""transformer_en-de"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_transformer_en_de_predict():
    """transformer_en-de"""
    os.system("hub install transformer_en-de")
    model = hub.Module(name="transformer_en-de", beam_size=5)
    src_texts = [
        "What are you doing now?",
        "The change was for the better; I eat well, I exercise, I take my drugs.",
        "Such experiments are not conducted for ethical reasons.",
    ]
    n_best = 3  # 每个输入样本的输出候选句子数量
    trg_texts = model.predict(src_texts, n_best=n_best)
    for idx, st in enumerate(src_texts):
        print("-" * 30)
        print(f"src: {st}")
        for i in range(n_best):
            print(f"trg[{i+1}]: {trg_texts[idx*n_best+i]}")
    os.system("hub uninstall transformer_en-de")
