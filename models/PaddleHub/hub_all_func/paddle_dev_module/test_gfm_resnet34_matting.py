"""gfm_resnet34_matting"""
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


def test_gfm_resnet34_matting_predict():
    """gfm_resnet34_matting predict"""
    os.system("hub install gfm_resnet34_matting")
    model = hub.Module(name="gfm_resnet34_matting")
    results = model.predict(["face_01.jpeg"])
    print(results)
    os.system("hub uninstall gfm_resnet34_matting")
