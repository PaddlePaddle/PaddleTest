"""ginet_resnet50vd_cityscapes"""
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


def test_ginet_resnet50vd_cityscapes_predict():
    """ginet_resnet50vd_cityscapes predict"""
    os.system("hub install ginet_resnet50vd_cityscapes")
    model = hub.Module(name="ginet_resnet50vd_cityscapes")
    img = cv2.imread("doc_img.jpeg")
    result = model.predict(images=[img], visualization=True)
    print(result)
    os.system("hub uninstall ginet_resnet50vd_cityscapes")
