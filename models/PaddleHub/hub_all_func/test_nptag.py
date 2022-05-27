"""nptag"""
import os
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_nptag_predict():
    """nptag predict"""
    os.system("hub install nptag")
    # Load NPTag
    module = hub.Module(name="nptag")
    # String input
    results = module.predict("糖醋排骨")
    print(results)
    # List input
    results = module.predict(["糖醋排骨", "红曲霉菌"])
    print(results)
    os.system("hub uninstall nptag")
