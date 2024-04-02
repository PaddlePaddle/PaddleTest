"""modnet_resnet50vd_matting"""
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


def test_modnet_resnet50vd_matting_predict():
    """modnet_resnet50vd_matting predict"""
    os.system("hub install modnet_resnet50vd_matting")
    model = hub.Module(name="modnet_resnet50vd_matting")
    results = model.predict(["face_01.jpeg"])
    print(results)
    os.system("hub uninstall modnet_resnet50vd_matting")
