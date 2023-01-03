"""Rumor_prediction"""
import os
import paddle

import paddlehub as hub

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_Rumor_prediction_predict():
    """Rumor_prediction predict"""
    os.system("hub install Rumor_prediction")
    module = hub.Module(name="Rumor_prediction")

    test_texts = [
        "兴仁县今天抢小孩没抢走，把孩子母亲捅了一刀，看见这车的注意了，真事，车牌号辽HFM055！！！！！"
        "赶紧散播！ 都别带孩子出去瞎转悠了 尤其别让老人自己带孩子出去 太危险了 注意了！！！！辽HFM055北京现代朗动，在各学校门口抢小孩！！！110已经 证实！！全市通缉！！"
    ]
    results = module.Rumor(texts=test_texts, use_gpu=use_gpu)
    print(results)
    os.system("hub uninstall Rumor_prediction")
