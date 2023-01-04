"""simnet_bow"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_simnet_bow_predict():
    """simnet_bow predict"""
    os.system("hub install simnet_bow")
    simnet_bow = hub.Module(name="simnet_bow")
    # Data to be predicted
    test_text_1 = ["这道题太难了", "这道题太难了", "这道题太难了"]
    test_text_2 = ["这道题是上一年的考题", "这道题不简单", "这道题很有意思"]
    inputs = {"text_1": test_text_1, "text_2": test_text_2}
    results = simnet_bow.similarity(data=inputs, batch_size=2)
    print(results)
    os.system("hub uninstall simnet_bow")
