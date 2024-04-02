"""unified_transformer_12L_cn"""
import os
import paddle

# 非交互模式
import paddlehub as hub

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_unified_transformer_12L_cn_predict():
    """unified_transformer_12L_cn predict"""
    os.system("hub install unified_transformer_12L_cn")
    model = hub.Module(name="unified_transformer_12L_cn")
    data = [["你是谁？"], ["你好啊。", "吃饭了吗？"]]
    result = model.predict(data)
    print(result)
    os.system("hub uninstall unified_transformer_12L_cn")
