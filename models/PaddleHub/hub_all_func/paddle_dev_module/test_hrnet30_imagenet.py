"""hrnet30_imagenet"""
import os
import paddle
import paddlehub as hub

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_hrnet30_imagenet_predict():
    """hrnet30_imagenet predict"""
    os.system("hub install hrnet30_imagenet")
    model = hub.Module(name="hrnet30_imagenet")
    result = model.predict(["doc_img.jpeg"])
    print(result)
    os.system("hub uninstall hrnet30_imagenet")
