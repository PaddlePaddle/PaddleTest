"""dim_vgg16_matting"""
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


def test_dim_vgg16_matting_predict():
    """dim_vgg16_matting predict"""
    os.system("hub install dim_vgg16_matting")
    model = hub.Module(name="dim_vgg16_matting")
    results = model.predict(image_list=["face_01.jpeg"], trimap_list=["face_01.jpeg"])
    print(results)
    os.system("hub uninstall dim_vgg16_matting")
