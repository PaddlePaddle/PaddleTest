"""transformer_nist_wait_5"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_transformer_nist_wait_5_predict():
    """transformer_nist_wait_5"""
    os.system("hub install transformer_nist_wait_5")
    model = hub.Module(name="transformer_nist_wait_5")
    # 待预测数据（模拟同声传译实时输入）
    text = [
        "他",
        "他还",
        "他还说",
        "他还说现在",
        "他还说现在正在",
        "他还说现在正在为",
        "他还说现在正在为这",
        "他还说现在正在为这一",
        "他还说现在正在为这一会议",
        "他还说现在正在为这一会议作出",
        "他还说现在正在为这一会议作出安排",
        "他还说现在正在为这一会议作出安排。",
    ]
    for t in text:
        print("input: {}".format(t))
        result = model.translate(t)
        print("model output: {}\n".format(result))
    os.system("hub uninstall transformer_nist_wait_5")
