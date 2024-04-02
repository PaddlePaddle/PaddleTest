"""ernie-csc"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_ernie_csc_predict():
    """ernie-csc"""
    os.system("hub install ernie-csc")
    # Load ernie-csc
    module = hub.Module(name="ernie-csc")
    # String input
    results = module.predict("遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。")
    print(results)
    # List input
    results = module.predict(["遇到逆竟时，我们必须勇于面对，而且要愈挫愈勇，这样我们才能朝著成功之路前进。", "人生就是如此，经过磨练才能让自己更加拙壮，才能使自己更加乐观。"])
    print(results)
    os.system("hub uninstall ernie-csc")
