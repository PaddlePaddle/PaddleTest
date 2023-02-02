"""ginet_resnet50vd_ade20k"""
import os
import cv2
import paddlehub as hub
import paddle

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_ginet_resnet50vd_ade20k_predict():
    """ginet_resnet50vd_ade20k predict"""
    os.system("hub install ginet_resnet50vd_ade20k")
    model = hub.Module(name="ginet_resnet50vd_ade20k")
    img = cv2.imread("doc_img.jpeg")
    result = model.predict(images=[img], visualization=True)
    print(result)
    os.system("hub uninstall ginet_resnet50vd_ade20k")
