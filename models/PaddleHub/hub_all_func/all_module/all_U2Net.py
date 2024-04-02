"""U2Net"""
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


def test_U2Net_predict():
    """U2Net predict"""
    os.system("hub install U2Net")
    model = hub.Module(name="U2Net")
    result = model.Segmentation(
        images=[cv2.imread("doc_img.jpeg")],
        paths=None,
        batch_size=1,
        input_size=320,
        output_dir="U2Net_output",
        visualization=True,
    )
    print(result)
    os.system("hub uninstall U2Net")
