"""ID_Photo_GEN"""
import os
import cv2
import paddle

import paddlehub as hub

if paddle.is_compiled_with_cuda():
    paddle.set_device("gpu")
    use_gpu = True
else:
    paddle.set_device("cpu")
    use_gpu = False


def test_ID_Photo_GEN_predict():
    """ID_Photo_GEN"""
    os.system("hub install ID_Photo_GEN")
    model = hub.Module(name="ID_Photo_GEN")
    result = model.Photo_GEN(
        images=[cv2.imread("face_01.jpeg")],
        paths=None,
        batch_size=1,
        output_dir="output_ID_Photo_GEN",
        visualization=True,
        use_gpu=use_gpu,
    )
    print(result)
    os.system("hub uninstall ID_Photo_GEN")
